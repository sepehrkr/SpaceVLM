# Space Idea Retrieval & MCQ

This repository contains our experiments and approaches for handling **caption splitting** (affirmative + negative parts) across multiple tasks such as image retrieval, multiple-choice question answering, and text-to-image evaluation.

---

## 📂 Repository Structure

### Retrieval & MCQ Tasks
- **`space_idea_retrieval.py`**  
  Approach to the image retrieval task using split captions.  
- **`space_idea_mcq.py`**  
  Approach to the multiple-choice question (MCQ) task using split captions.  

### Retrieval & MCQ with LLM-based Splitting
- **`space_idea_retrieval_LLM.py`**  
  Approach to the image retrieval task using an LLM for caption splitting.  
- **`space_idea_mcq_LLM.py`**  
  Approach to the MCQ task using an LLM for caption splitting.  

### Baseline & Embedding Analysis
- **`negation.ipynb`**  
  Baseline experiments for handling negation.  
- **`embedding_space_divisability.ipynb`**  
  Analysis and justification of the divisibility of CLIP embedding space.  

### Other Tasks
- **`T2I.ipynb`**  
  Text-to-Image evaluation task.  
- **`finetuning_negation_LLM.py`**  
  Script for fine-tuning Mistral (or any other LLM) to split input captions into affirmative and negative parts.  

---
