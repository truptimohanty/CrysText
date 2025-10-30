import torch
import numpy as np
import os
from unsloth import FastLanguageModel


## Trained Model available on Huggingface
#truptimohanty/CrysText_Ehull_LLaMA
#truptimohanty/CrysText_Ehull_Mistral

# Define the destination folder
destination_folder = "generation_temp_1p2_mistral"  

# Configuration
max_seq_length = 2048 
dtype = None  
load_in_4bit = True  # Use 4-bit quantization for memory efficiency.



# Generate 1000 uniformly sampled values between 0.0 and 0.08
np.random.seed(42)
ehull_list = np.round(np.random.uniform(0.0, 0.08, 1000), 3)

# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="truptimohanty/CrysText_Ehull_Mistral", ## path to the trained model adapters available on Huggingface
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

# Enable faster inference
FastLanguageModel.for_inference(model)


os.makedirs(destination_folder, exist_ok=True)

# Check how many files have already been generated
existing_files = {f.split("_")[0] for f in os.listdir(destination_folder) if f.endswith(".cif")}

# Iterate over the ehull_list and generate CIF files
for i, ehull_value in enumerate(ehull_list):
    if str(i) in existing_files:
        print(f"Sample {i} already exists. Skipping...")
        continue

    file_path = os.path.join(destination_folder, f"{i}_ehull_{ehull_value}_gen_llama_4bit_mp20_unconditional.cif")

    # Prepare the input for the model
    user_input = f"Energy above hull = {ehull_value} eV/atom."

    try:
        inputs = tokenizer(
            [
                alpaca_prompt.format(
                    "Generate CIF for a stable material based on the given description.",  # Instruction
                    user_input,
                    ""  # Leave blank for generation
                )
            ],
            return_tensors="pt"
        ).to("cuda")

        # Generate the output
        outputs = model.generate(**inputs, max_new_tokens=3000,do_sample=True, temperature=1.2, use_cache=True)
        raw_output = tokenizer.batch_decode(outputs)

        # Extract the response and clean the CIF content
        response_start = raw_output[0].split("### Response:")[1]
        cleaned_cif_content = response_start.strip("</s>").strip() 

        # Save the cleaned CIF content to the destination folder
        with open(file_path, "w") as f:
            f.write(cleaned_cif_content)

        print(f"Generated and saved CIF for sample {i}.")

    except Exception as e:
        print(f"Failed to generate CIF for sample {i}: {e}")
