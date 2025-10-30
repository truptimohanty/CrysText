import os
import numpy as np
import itertools
import shutil
from collections import Counter
from pymatgen.core import Structure
from smact import element_dictionary, metals, neutral_ratios
from smact.screening import pauling_test
import json
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from tqdm import tqdm


## Check for valid generations
# Input and output folder paths
input_folder = "generation_cifs"  
output_folder = "valid_generation_cifs"

def get_safe_space_group_number(structure):
    '''
    Symmetry check
    '''
    try:
        spg_info = structure.get_space_group_info()
        return spg_info[1] if spg_info else None
    except Exception:
        return None
    
def structure_validity(structure, cutoff=0.5):
    # Distance and volume checks
    dist_mat = structure.distance_matrix
    dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (cutoff + 10.))
    if dist_mat.min() < cutoff or structure.volume < 0.1:
        return False

    # Symmetry check
    spg_number = get_safe_space_group_number(structure)
    if spg_number is None:
        return False  # Symmetry undefined, reject structure

    return True


# Compositional (SMACT) validity check
# adapted from
# https://github.com/lantunes/CrystaLLM/blob/main/bin/benchmark_metrics.py
def smact_validity(atom_types, use_pauling_test=True, include_alloys=True):
    elem_counter = Counter(atom_types)
    elems = [(elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())]
    comp, elem_counts = list(zip(*elems))
    elem_counts = np.array(elem_counts)
    elem_counts = elem_counts / np.gcd.reduce(elem_counts)
    count = tuple(elem_counts.astype("int").tolist())

    elem_symbols = tuple(comp)
    space = element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]

    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    oxn = 1
    for oxc in ox_combos:
        oxn *= len(oxc)
    if oxn > 1e7:
        return False

    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        cn_e, cn_r = neutral_ratios(ox_states, stoichs=stoichs, threshold=threshold)
        if cn_e:
            if use_pauling_test:
                electroneg_OK = pauling_test(ox_states, electronegs)
            else:
                electroneg_OK = True
            if electroneg_OK:
                return True
    return False

# Extract atom types from structure
def get_atom_types_from_structure(structure):
    return [str(element) for element in structure.species]


# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Counters
total = 0
structure_valid_count = 0
composition_valid_count = 0
both_valid_count = 0

# Loop through CIFs
for filename in os.listdir(input_folder):
    if filename.endswith(".cif"):
        total += 1
        full_path = os.path.join(input_folder, filename)
        try:
            structure = Structure.from_file(full_path)
            atom_types = get_atom_types_from_structure(structure)

            struct_valid = structure_validity(structure)
            comp_valid = smact_validity(atom_types)

            if struct_valid:
                structure_valid_count += 1
            if comp_valid:
                composition_valid_count += 1
            if struct_valid and comp_valid:
                both_valid_count += 1
                shutil.copy(full_path, os.path.join(output_folder, filename))
        except Exception as e:
            print(f"Error processing {filename}: {e}")


# Summary
print("\n=== Validity Summary ===")
print(f"Total CIFs checked:              {total}")
print(f"Structural validity passed:     {structure_valid_count}")
print(f"Compositional validity passed:  {composition_valid_count}")
print(f"Both valid (saved):             {both_valid_count}")
print(f"Valid CIFs saved to:          {output_folder}")


## Check for unique and novel generations
generated_folder = "valid_generation_cifs/"
training_folder = "mp20_all_cifs/"
unique_folder = "unique_structures"
novel_folder = "novel_structures"

os.makedirs(unique_folder, exist_ok=True)
os.makedirs(novel_folder, exist_ok=True)

# Helper to load structures from a folder
def load_structures(folder):
    structs = []
    for filename in os.listdir(folder):
        if filename.endswith(".cif"):
            try:
                struct = Structure.from_file(os.path.join(folder, filename))
                structs.append((filename, struct))
            except:
                continue
    return structs

# Load structures
generated = load_structures(generated_folder)
train_structures = [s for _, s in load_structures(training_folder)]

# Stage 1: Deduplicate generated (uniqueness check)
print("Checking uniqueness...")
matcher = StructureMatcher()
unique_structures = []

for fname, gen_struct in tqdm(generated, desc="Filtering unique structures"):
    is_duplicate = False
    for _, existing_struct in unique_structures:
        if matcher.fit(gen_struct, existing_struct):
            is_duplicate = True
            break
    if not is_duplicate:
        gen_struct.to(filename=os.path.join(unique_folder, fname))
        unique_structures.append((fname, gen_struct))

print(f"Unique structures saved: {len(unique_structures)}")

# Stage 2: Check novelty vs training set
print("Checking novelty...") ## To do use composition
novel_count = 0
for fname, unique_struct in tqdm(unique_structures, desc="Filtering novel structures"):
    is_novel = True
    for train_struct in train_structures:
        if matcher.fit(unique_struct, train_struct):
            is_novel = False
            break
    if is_novel:
        unique_struct.to(filename=os.path.join(novel_folder, fname))
        novel_count += 1

print(f"Novel structures saved: {novel_count}/{len(unique_structures)}")