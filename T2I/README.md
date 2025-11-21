# Text-to-Image (T2I) Task Documentation

This directory contains the datasets and scripts required to perform the **Text-to-Image (T2I)** task. Please clone the GALIP repo (https://github.com/tobran/GALIP).

---

## 📂 `prompts_with_questions.json`  [[Park et al., 2025]](#reference)

**Description:**  
Contains a collection of prompts and their corresponding questions used to evaluate model performance in generating **affirmative** and excluding **negative** objects.

---

## 📂 `prompts_with_questions_multiple_neagation.json` [created by us]

**Description:**  
Contains a collection of prompts and their corresponding questions used to evaluate model performance in a multi-negation setup for generating **affirmative** and excluding **negative** objects.

---

## 📄 `splitted_prompts.csv`

**Description:**  
Derived from `prompts_with_questions.json` by splitting each prompt into its **affirmative** and **negative** counterparts.  
This dataset is used within the main T2I evaluation notebook.

---

## 💻 `T2I.ipynb`

**Description:**  
A Jupyter notebook for generating and evaluating our proposed approach compared to baseline models:  
`CLIP`, `CLIP-NegFull`, `ConCLIP`, `NegCLIP`, and `NegCLIP-NegFull`.


---

## 📚 Reference

```bibtex
@article{park2025know,
  title={Know "No" Better: A Data-Driven Approach for Enhancing Negation Awareness in CLIP},
  author={Park, Junsung and Lee, Jungbeom and Song, Jongyoon and Yu, Sangwon and Jung, Dahuin and Yoon, Sungroh},
  journal={arXiv preprint arXiv:2501.10913},
  year={2025}
}
```
