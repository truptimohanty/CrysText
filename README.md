# CrysText: A Generative AI Approach for Text-Conditioned Crystal Structure Generation using LLM 


The ability to generate crystal structures directly from textual descriptions marks a pivotal advancement in materials informatics and underscores the emerging role of large language models (LLMs) in inverse design. In this work, we introduce CrysText, a text-conditioned framework that generates crystal structures in Crystallographic Information File (CIF) format from natural language prompts specifying composition and space group. Leveraging LLaMA-3.1-8B and Mistral-7B-v0.3 fine-tuned using Quantized Low-Rank Adaptation (QLoRA), our approach enables the efficient and scalable generation of CIF-formatted structures directly from input descriptions, eliminating the need for post-processing with rapid inference. Evaluations on the MP-20 benchmark demonstrate high structural match rates and low RMSE values, confirming the model’s ability to generate physically consistent crystal structures aligned with compositional and symmetry constraints. By incorporating energy above the convex hull as a conditioning parameter, CrysText further demonstrates the ability to generate thermodynamically stable novel materials. 

We subsequently extend this framework with CrysText-RL, which integrates Group Relative Policy Optimization (GRPO) to provide reinforcement learning feedback directly on generated CIF outputs via group-based normalized rewards. CrysText-RL achieves additional improvements over the supervised CrysText model in terms of composition and space group satisfiability and structure match rate. This work establishes a scalable paradigm for text-driven crystal structure generation, demonstrating that both supervised fine-tuning and reinforcement learning enable a pathway towards accelerated materials discovery.

## Crystal Structure Prediction
![overview](images/CrysText_CSP_final.gif)

## Crystal Structure Generation
![Ehull](images/CrysText_ehull_final.gif)

## CrysText-RL workflow

![CrysText-RL](images/CrysText-RL_workflow.png)

#### Workflow of CrysText-RL for crystal structure prediction using Group Relative Policy Optimization. The training phase begins with a textual prompt specifying the target composition and space group, which is provided to the policy model. The model generates multiple candidate crystal structures in CIF format (n=6 samples per prompt). Each generated structure undergoes evaluation through a multi-stage reward function comprising: (i) CIF format validation by parsing the generated output using pymatgen to obtain structure object, (ii) structural validity assessment based on interatomic distances and unit cell volume, (iii) compositional accuracy verification against the target composition, and (iv) structural match evaluation against the ground-truth structure using the pymatgen StructureMatcher class. Rewards from all generated samples are normalized using the group mean and standard deviation to compute advantage values, which drive policy updates under the GRPO loss function. The CrysText reference model is used to compute the KL divergence term, constraining policy updates to prevent excessive deviation from the reference model.


