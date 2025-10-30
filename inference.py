from unsloth import FastLanguageModel
import torch
import argparse
import os
from pymatgen.core import Composition

## Trained Model available on Huggingface
#truptimohanty/CrysText
#truptimohanty/CrysText_RL


# Argument parsing for user input
parser = argparse.ArgumentParser(description="Generate CIF file from model.")
parser.add_argument('model_path', type=str, nargs='?', default='truptimohanty/CrysText',help='Path to the fine-tuned model')
parser.add_argument('composition', type=str,  nargs='?', default='NdAgHg2',help='Material composition (e.g., NdAgHg2)')
parser.add_argument('space_group', type=int,  nargs='?', default=225,help='Space group number (e.g., 225)')
parser.add_argument('cif_save_path', nargs='?', default='./generated_structure.cif', help='Path to save the generated CIF file (default: current directory)')
args = parser.parse_args()


# This code uses the Unsloth library for fine-tuning
# Unsloth: https://github.com/unslothai/unsloth


# Configuration
max_seq_length = 2048
dtype = None
load_in_4bit = True

# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.model_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Define the Alpaca prompt template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Enable the inference
FastLanguageModel.for_inference(model)
comp = Composition(args.composition)


# Prepare the input for the model
user_input = f"Material composition is {comp.reduced_formula}. It has a space group number {args.space_group}."
print(user_input)
# " It has a space group number {args.space_group}."
inputs = tokenizer(
    [
        alpaca_prompt.format(
            "Generate CIF for the given material description",  # Instruction
            user_input,
            ""  # Leave blank for generation
        )
    ], return_tensors="pt"
).to("cuda")

# Generate the output
outputs = model.generate(**inputs, max_new_tokens=3000, temperature=1.0,do_sample = True, use_cache=True)
raw_output = tokenizer.batch_decode(outputs)

# Extract the response and clean the CIF content
response_start = raw_output[0].split("### Response:")[-1]

cleaned_cif_content = response_start.strip("</s>").strip() # use "<|end_of_text|>" instead of "</s>" for LLaMa model

# Save the cleaned CIF content to the specified path
with open(args.cif_save_path, "w") as f:
    f.write(cleaned_cif_content)

print(f"Generated and saved CIF at {args.cif_save_path}")
