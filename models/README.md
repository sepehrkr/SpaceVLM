# 📁 Baseline Models Directory

This directory contains baseline models other than CLIP, including **CLIP-NegFull**, **ConCLIP**, **NegCLIP**, and **NegCLIP-NegFull**.  
Each model and its checkpoint are listed below, along with a short description and corresponding paper reference.

---

## 🧩 Models Overview

### 1. ConCLIP (WACV 2025)
- **File:** `conclip_b32_openclip_version.pt`  
- **Description:** A contrastively trained CLIP model designed to improve negation understanding.  
- **Paper:** [Original Paper Link](#)  

---

### 2. NegCLIP (ICLR 2023)
- **File:** `negclip.pth`  
- **Description:** An improved version of CLIP, fine-tuned for enhanced compositional language understanding.  
- **Paper:** [Original Paper Link](#)  

---

### 3. NegCLIP_CC12M_NegFull (CVPR 2025)
- **File:** `NegCLIP_CC12M_NegFull_ViT-B-32_lr1e-8_clw0.99_mlw0.01.pt`  
- **Description:** A fine-tuned version of NegCLIP trained on the **CC12M-NegFull** dataset, which combines **CC12M-NegCap** and **CC12M-NegMCQ**.  
  This is our **best-performing model** in terms of negation understanding.  
- **Paper:** [Original Paper Link](#)  

---

### 4. CLIP_CC12M_NegFull (CVPR 2025)
- **File:** `CLIP_CC12M_NegFull_ViT-B-32_lr1e-8_clw0.99_mlw0.01.pt`  
- **Description:** A CLIP model initialized with OpenAI weights and fine-tuned on the **CC12M-NegFull** dataset.  
- **Paper:** [Original Paper Link](#)  

---

## 📦 Download Instructions

All the above model checkpoints can be downloaded from the following Google Drive folder:  
🔗 [Google Drive Link](https://drive.google.com/drive/folders/1kSEq0mkV1t1T8GuOAM65iz_iAA7e5gxB?usp=sharing)

After downloading, place the model files in the directory structure below:
```
models/
├── ConCLIP/
│ └── conclip_b32_openclip_version.pt
├── NegCLIP/
│ └── negclip.pth
├── CLIP_CC12M_NegFull/
│ └── checkpoint.pt
├── NegCLIP_CC12M_NegFull/
│ └── checkpoint.pt
```
