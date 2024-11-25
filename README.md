# CrysText: Advancing Text-Conditioned Crystal Structure Generation through Fine-Tuned Large Language Model


Generating crystal structures directly from textual descriptions marks a pivotal advancement in materials informatics, offering a streamlined pathway from concept to discovery. Integrating generative models into Crystal Structure Prediction (CSP) presents a transformative opportunity to enhance efficiency and innovation. While large language models (LLMs) excel at understanding and generating text, their potential in materials discovery remains largely unexplored. Here, we introduce CrysText, an advanced approach for generating crystal structures from simple text prompts, conditioned on material composition and space group number. Leveraging Llama-3.1-8B fine-tuned with Quantized Low-Rank Adapters (QLoRA), our approach enables efficient and scalable generation of CIF-formatted structures directly from input descriptions without post-processing. This method significantly reduces computational costs during fine-tuning and ensures rapid inference. Evaluations on the MP-20 benchmark dataset demonstrate high structure match rates and effective RMS metrics, showcasing the framework's ability to generate crystal structures that faithfully adhere to specified compositions and crystal symmetries. By successfully conditioning on energy above the hull, we further demonstrate the potential of CrysText to generate stable crystal structures. Our work highlights the transformative role of LLMs in text-prompted inverse design, accelerating the discovery of new materials.

Releasing Code Soon!

![Screenshot](images/CrysText.png)


