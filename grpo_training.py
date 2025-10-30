
import os
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from transformers import TrainingArguments,TextStreamer
from trl import SFTTrainer, GRPOConfig, GRPOTrainer
from vllm import SamplingParams
from pymatgen.core import Structure, Element
from pymatgen.analysis.structure_matcher import StructureMatcher


# This code uses the Unsloth library for fine-tuning
# Unsloth: https://github.com/unslothai/unsloth

max_seq_length = 2048
dtype = None
load_in_4bit = True 
lora_rank = 16
file_path = "train.csv"
output_dir = "grpo_checkpoint"
lora_adapetr_path = "trained_lora_adapters"

model, tokenizer = FastLanguageModel.from_pretrained(
    
    model_name = "truptimohanty/CrysText", ## fine-tuned mistral model available of Huggingface
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    max_lora_rank = lora_rank,
    fast_inference = True,
    gpu_memory_utilization = 0.6


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def formatting_grpo_prompts_func(examples):
   '''This function reformats dataset rows into **prompt–answer pairs** for GRPO training.
   It combines each instruction, input, and output using a template alpaca_prompt
   and returns lists of prompts and answers.
   '''
   instructions = examples["instruction"]
   inputs       = examples["input"]
   targets      = examples["output"]
   outputs=""
   prompt = []
   answers = []
   for instruction, input, target in zip(instructions, inputs, targets):

     text = alpaca_prompt.format(instruction,input,outputs)
     prompt.append(text)
     answers.append(target)

   return { "prompt" : prompt,
            "answer": answers}

dataset = Dataset.from_pandas(pd.read_csv(file_path))
dataset = dataset.map(formatting_grpo_prompts_func, batched=True)


### Functions to assign the rewards

def get_safe_space_group_number(structure):
   '''
   Attempts to retrieve the space group number of a crystal structure.

   Parameters
   ----------
   structure : pymatgen.core.Structure
       A Pymatgen structure object.

   Returns
   -------
   int or float
       The space group number (int) if it can be determined.
       Returns numpy.nan if space group information cannot be retrieved.
   '''
   try:
       spg_info = structure.get_space_group_info()
       return spg_info[1] if spg_info else np.nan
   except Exception:
       return np.nan


def structure_validity(structure, cutoff=0.5):
   """
   Check if a crystal structure is valid based on interatomic distances,
   volume, and ability to assign a space group.

   Parameters
   ----------
   structure : pymatgen.core.Structure
       The crystal structure to check.
   cutoff : float, optional (default=0.5)
       Minimum allowed interatomic distance in Ångströms.
       If atoms are closer than this, the structure is invalid.

   Returns
   -------
   bool
       True if the structure is valid, False otherwise.
   """
   dist_mat = structure.distance_matrix


   dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (cutoff + 10.))
   if dist_mat.min() < cutoff or structure.volume < 0.1:
       return False
   spg_number = get_safe_space_group_number(structure)
   return not pd.isna(spg_number)


def compare_structures(struct1, struct2, stol=0.5, ltol=0.3, angle_tol=10):

   """
   Compare two crystal structures to check if they are equivalent.

   Parameters
   ----------
   struct1 : pymatgen.core.Structure
       First structure.
   struct2 : pymatgen.core.Structure
       Second structure.
   stol : float, optional (default=0.5)
       Site tolerance in Å. Controls how close atomic positions must be
       to consider sites equivalent.
   ltol : float, optional (default=0.3)
       Lattice parameter tolerance (fractional). Controls how much
       lattices can differ in lengths and still be considered equivalent.
   angle_tol : float, optional (default=10)
       Angle tolerance in degrees. Controls how much unit cell angles
       can differ and still be considered equivalent.

   Returns
   -------
   bool
       True if the two structures are considered equivalent, False otherwise.
   """

   matcher = StructureMatcher(stol=stol, ltol=ltol, angle_tol=angle_tol)

   try:
       rms_result = matcher.get_rms_dist(struct1, struct2)

       if rms_result is not None:
           return True
       return False
   except:
       return False


def validate_structure(completion, answer):
   '''
   Parameters
   ----------
   completion : str
       The generated CIF string to validate.
   answer : str
       The reference CIF string (ground truth).

   Returns
   -------
   float
       A reward score based on parsing success, validity checks,
       composition match, and structural similarity.
       Returns -5 if parsing fails.
   '''
   reward = 0
   cleaned_cif_content = completion
   actual = Structure.from_str(answer, fmt="cif")

   try:

       valid_structure = Structure.from_str(cleaned_cif_content, fmt="cif")
       reward = 0.5  # reward for valid structure parsing
       # print("It passees pymatgen struct test",reward)

       if structure_validity(valid_structure):
           reward += 0.5
           # print("It passed inter atomic dist  test",reward)

       if valid_structure.composition.reduced_formula == actual.composition.reduced_formula:
           # print(valid_structure.composition.reduced_formula)
           # print(actual.composition.reduced_formula)
           reward += 0.5
           # print("It passed composition test",reward)


       rms_dist_highTol = compare_structures(actual, valid_structure,stol=0.9,ltol=0.7,angle_tol=20)

       if rms_dist_highTol:
           reward += 0.25
           # print("It passed match highTol",reward)

       rms_dist_midTol = compare_structures(actual, valid_structure,stol=0.7,ltol=0.5,angle_tol=15)

       if rms_dist_midTol:
           reward += 0.25
           # print("It passed match midTol",reward)

       rms_dist_lowTol = compare_structures(actual, valid_structure,stol=0.5, ltol=0.3, angle_tol=10)

       if rms_dist_lowTol:
           reward += 1.0
           # print("It passed match lowTol",reward)

       return reward

   except Exception as e:
       # Uncomment the next line for debugging if you want:
       # print("Exception caught in validate_structure:", e)
       return -2




def reward_func(prompts, completions, answer, **kwargs):

# print([validate_structure(completion,ans) for completion,ans in zip(completions,answer)])

 return [validate_structure(completion,ans) for completion,ans in zip(completions,answer)]


max_prompt_length = 100
vllm_sampling_params = SamplingParams(
   min_p = 0.1,
   top_p = 1.0,
   top_k = -1,
   seed = 3407,
   stop = [tokenizer.eos_token],
   include_stop_str_in_output = True,
)



training_args = GRPOConfig(
   vllm_sampling_params = vllm_sampling_params,
   temperature = 1.0,
   learning_rate = 1e-6,
   adam_beta1 = 0.9,
   adam_beta2 = 0.99,
   weight_decay = 0.1,
   warmup_ratio = 0.1,
   lr_scheduler_type = "linear",
   optim = "paged_adamw_8bit",
   logging_steps = 1,
   per_device_train_batch_size = 12,
   gradient_accumulation_steps = 1,
   num_generations = 6,
   max_prompt_length = max_prompt_length,
   max_completion_length = max_seq_length - max_prompt_length,
   num_train_epochs = 1, # Set to 1 for a full training run
   save_steps = 500,
   max_grad_norm = 0.1,
   report_to = "wandb", # Can use Weights & Biases
   output_dir = outpu_dir,
   # report_to= None
)

trainer = GRPOTrainer(
   model = model,
   processing_class = tokenizer,
   reward_funcs = reward_func,
   args = training_args,
   train_dataset = dataset,
)
#trainer.train(resume_from_checkpoint = True)
trainer.train()
model.save_pretrained(lora_adapetr_path) # Local saving
tokenizer.save_pretrained(lora_adapetr_path)


