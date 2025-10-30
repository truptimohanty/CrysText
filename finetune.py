
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
import os
from datasets import Dataset
import pandas as pd
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from transformers import TextStreamer


# This code uses the Unsloth library for fine-tuning
# Unsloth: https://github.com/unslothai/unsloth

max_seq_length = 2048 
dtype = None 
load_in_4bit = True 
lora_rank = 16
## provide the path of the training data consisting of instuction, input and output
file_path = "train.csv"

## path to save the trained lora adapter
lora_adapetr_path = "trained_lora_adapters"

## path to save the check poinnts
output_dir = "finetune_checkpoint"
num_train_epochs = 10

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-v0.3",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit
)



model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = lora_rank,
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
    use_rslora = False,  
    loftq_config = None,
)



prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass



# Load the dataset from a CSV file
dataset = Dataset.from_pandas(pd.read_csv(file_path))

# Apply the formatting function to the dataset
dataset = dataset.map(formatting_prompts_func, batched=True)

# Check the first formatted example
print(dataset[0])



trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        output_dir = output_dir,
        save_strategy = "steps",
        save_steps = 424,
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 10, # Set this for 1 full training run.
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        report_to =  "wandb",
	    weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,

    ),
)



trainer_stats = trainer.train()
#trainer_stats = trainer.train(resume_from_checkpoint = True)


model.save_pretrained(lora_adapetr_path) # Local saving
tokenizer.save_pretrained(lora_adapetr_path)

