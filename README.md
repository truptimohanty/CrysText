# CrysText: A Generative AI Approach for Text-Conditioned Crystal Structure Generation using LLM

The ability to generate crystal structures directly from textual descriptions marks a pivotal advancement in materials informatics and underscores the emerging role of large language models (LLMs) in inverse design. In this work, we introduce CrysText, a text-conditioned framework that generates crystal structures in Crystallographic Information File (CIF) format from natural language prompts specifying composition and space group. Leveraging LLaMA-3.1-8B and Mistral-7B-v0.3 fine-tuned using Quantized Low-Rank Adaptation (QLoRA), our approach enables the efficient and scalable generation of CIF-formatted structures directly from input descriptions, eliminating the need for post-processing with rapid inference. Evaluations on the MP-20 benchmark demonstrate high structural match rates and low RMSE values, confirming the model's ability to generate physically consistent crystal structures aligned with compositional and symmetry constraints. By incorporating energy above the convex hull as a conditioning parameter, CrysText further demonstrates the ability to generate thermodynamically stable novel materials. 

We subsequently extend this framework with CrysText-RL, which integrates Group Relative Policy Optimization (GRPO) to provide reinforcement learning feedback directly on generated CIF outputs via group-based normalized rewards. CrysText-RL achieves additional improvements over the supervised CrysText model in terms of composition and space group satisfiability and structure match rate. This work establishes a scalable paradigm for text-driven crystal structure generation, demonstrating that both supervised fine-tuning and reinforcement learning enable a pathway towards accelerated materials discovery.

## Crystal Structure Prediction
![overview](images/CrysText_CSP_final.gif)

## Crystal Structure Generation
![Ehull](images/CrysText_ehull_final.gif)

## CrysText-RL workflow

![CrysText-RL](images/CrysText-RL_workflow.png)

#### Workflow of CrysText-RL for crystal structure prediction using Group Relative Policy Optimization. The training phase begins with a textual prompt specifying the target composition and space group, which is provided to the policy model. The model generates multiple candidate crystal structures in CIF format (n=6 samples per prompt). Each generated structure undergoes evaluation through a multi-stage reward function comprising: (i) CIF format validation by parsing the generated output using pymatgen to obtain structure object, (ii) structural validity assessment based on interatomic distances and unit cell volume, (iii) compositional accuracy verification against the target composition, and (iv) structural match evaluation against the ground-truth structure using the pymatgen StructureMatcher class. Rewards from all generated samples are normalized using the group mean and standard deviation to compute advantage values, which drive policy updates under the GRPO loss function. The CrysText reference model is used to compute the KL divergence term, constraining policy updates to prevent excessive deviation from the reference model.

---

## Installation

### Prerequisites
- Python 3.11
- CUDA 12.6 compatible GPU
- Conda or Miniconda

### Setup Environment

1. **Create a new conda environment:**
```bash
conda create -n CrysText python=3.11 -y
conda activate CrysText
```

2. **Install dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

### Inference

Generate crystal structures from composition and space group:

```bash
python inference.py [model_path] [composition] [space_group] [output_cif_path]
```

**Example:**
```bash
python inference.py truptimohanty/CrysText NdAgHg2 225 output.cif
```

**Default values:**
- `model_path`: `truptimohanty/CrysText`
- `composition`: `NdAgHg2`
- `space_group`: `225`
- `output_cif_path`: `./generated_structure.cif`

### Fine-tuning

Train the model on your own dataset:

**Arguments:**
- `--model`: Base model (default: `unsloth/mistral-7b-v0.3`)
- `--train_csv`: Path to training CSV with columns: `instruction`, `input`, `output`
- `--output_dir`: Checkpoint directory (default: `finetune_checkpoint`)
- `--adapters_dir`: LoRA adapters directory (default: `trained_lora_adapters`)
- `--max_seq_length`: Maximum sequence length (default: `2048`)
- `--lora_rank`: LoRA rank (default: `16`)
- `--num_train_epochs`: Number of epochs (default: `10`)

### GRPO Training

Train with reinforcement learning:

```bash
python grpo_training.py
```

---

## Pre-trained Models

All models are available on Hugging Face:

### CrysText Models
- **[truptimohanty/CrysText](https://huggingface.co/truptimohanty/CrysText)** - Supervised fine-tuned Mistral-7B-v0.3 model for crystal structure generation
- **[truptimohanty/CrysText_RL](https://huggingface.co/truptimohanty/CrysText_RL)** - GRPO-trained model with reinforcement learning feedback

### E_hull Conditioning Models
- **[truptimohanty/CrysText_Ehull_Mistral](https://huggingface.co/truptimohanty/CrysText_Ehull_Mistral)** - Mistral-7B-v0.3 conditioned on energy above convex hull
- **[truptimohanty/CrysText_Ehull_LLaMA](https://huggingface.co/truptimohanty/CrysText_Ehull_LLaMA)** - LLaMA-3.1-8B conditioned on energy above convex hull

### Usage in Code
```python
# Load CrysText model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="truptimohanty/CrysText",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True
)

# Load CrysText-RL model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="truptimohanty/CrysText_RL",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True
)

# Load E_hull Mistral model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="truptimohanty/CrysText_Ehull_Mistral",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True
)

# Load E_hull LLaMA model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="truptimohanty/CrysText_Ehull_LLaMA",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True
)
```

---

## 🚀 Quick Setup Guide

Get started in 3 minutes:

```bash
# 1. Create environment
conda create -n CrysText python=3.11 -y
conda activate CrysText

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Run inference
python inference.py

python inference.py truptimohanty/CrysText "NdAgHg2" 225 output.cif

# 4. Fine-tune (optional, default: mistral-7b-v0.3)
python finetune.py

# 5. Conditioning with E_hull
python conditioning_ehull.py
```

**Usage:** 
- Inference: `python inference.py [model_path] [composition] [space_group] [output_file]`
- Fine-tune: `python finetune.py` (uses `unsloth/mistral-7b-v0.3` by default)
- Conditioning: `python conditioning_ehull.py`

---

