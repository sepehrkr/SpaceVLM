# Vision-Language Models Can Understand Negation

This repository contains the code accompanying the paper **"Vision-Language Models Can Understand Negation"**.  

---

## 📂 Repository Structure

### `./data`
- Contains the datasets required to run the scripts.  
- Please refer to the `README.md` file in the `./data` directory for instructions on downloading the necessary datasets.  

### `./Retrieval`
- Contains code for running the **Retrieval** experiments described in the paper.  

### `./MCQ`
- Contains code for running the **Multiple-Choice Question (MCQ)** experiments.  

### `./T2I`
- Contains code for running the **Text-to-Image (T2I)** experiments.  

### `finetuning_negation_LLM.py`
- A script for supervised fine-tuning (SFT) of LLMs (e.g., Mistral).  
- The model learns to split an input caption into its **affirmative** and **negative** parts.  
  - Example:  
    - Input: *"A photo of a dog but not on grass"*  
    - Output: *("A photo of a dog", "A photo of grass")*  
- To run the script, use the following command:  
  ```bash
  accelerate launch --model_name mistralai/Mistral-7B-Instruct-v0.1
