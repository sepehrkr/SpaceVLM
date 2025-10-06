import clip
import torch
import numpy as np
import pandas as pd
import ast
from transformers import CLIPTokenizer
from torch.utils.data import Dataset, DataLoader, default_collate, Subset, random_split
from tqdm import tqdm
import datasets
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from video_utils.frame_sampler import UniformFrameSampler
from video_utils.video_reader import VideoReader
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import seaborn as sns
import os
import argparse 


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def process_MCQ_caption(s: str):
    aff = ""
    neg = ""

    if " not " in s:
        if " does " in s:
            neg = s.split(" does not include ")[1]

        else:
            neg = s.split(" not ")[1]

    
    if " does not " not in s:
        if " but " in s:
            aff_cand = s.split(" but ")[0]
            aff = aff_cand.split(" includes ")[1]
        else:
            try:
                aff = s.split(" includes ")[1]
            except:
                Exception("Error last ....")


    return aff, neg


def preprocess_retreival_caption(cap: str):

    pos_cap = ""
    neg_cap = ""

    idx = cap.find('There is no ')
    assert idx != -1
    
    if idx == 0:
        neg_cap = cap.split('. ')[0]
        neg_cap = neg_cap.split('There is no ')[1]
        neg_cap = neg_cap.split(' in the image')[0]

        pos_cap = '. '.join(cap.split('. ')[1:])
    
    else:
        neg_cap = cap[idx:]
        neg_cap = neg_cap.split('There is no ')[1]
        neg_cap = neg_cap.split(' in the image')[0]

        pos_cap = cap[:idx]

    return pos_cap.strip(), neg_cap.strip()




def main():
    
    parser = argparse.ArgumentParser()  

    parser.add_argument("--root", type=str, default='/home/amirhossein.hajimohammadrezaie/ECOR_extended/Negation')
    parser.add_argument("--repharased_MCQ_csv_path", type=str, default='data/images/VOC2007_mcq_llama3.1_rephrased.csv')
    parser.add_argument("--templated_MCQ_csv_path", type=str, default='data/images/VOC2007_mcq.csv')
    parser.add_argument("--repharased_Retreival_csv_path", type=str, default='data/images/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv')
    parser.add_argument("--templated_Retreival_csv_path", type=str, default='data/images/COCO_val_negated_retrieval_template.csv')
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--accumulation_steps", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=10)

    args = parser.parse_args()
    os.chdir(args.root)
    set_seed(args.seed)
    
    # Load Dataset
    mcq_repharased_df = pd.read_csv(args.repharased_MCQ_csv_path)
    mcq_repharased_df['repharased_caption'] = mcq_repharased_df.apply(lambda row: [row['caption_0'], row['caption_1'], row['caption_2'], row['caption_3']], axis=1)
    mcq_repharased_caps = mcq_repharased_df['repharased_caption'].explode(ignore_index=True)

    mcq_templated_df = pd.read_csv(args.templated_MCQ_csv_path)
    mcq_templated_df['caption_0'] = mcq_templated_df['caption_0'].apply(lambda s: f"### Positive Part\n{process_MCQ_caption(s)[0]}\n\n### Negative Part\n{process_MCQ_caption(s)[1]}")
    mcq_templated_df['caption_1'] = mcq_templated_df['caption_1'].apply(lambda s: f"### Positive Part\n{process_MCQ_caption(s)[0]}\n\n### Negative Part\n{process_MCQ_caption(s)[1]}")
    mcq_templated_df['caption_2'] = mcq_templated_df['caption_2'].apply(lambda s: f"### Positive Part\n{process_MCQ_caption(s)[0]}\n\n### Negative Part\n{process_MCQ_caption(s)[1]}")
    mcq_templated_df['caption_3'] = mcq_templated_df['caption_3'].apply(lambda s: f"### Positive Part\n{process_MCQ_caption(s)[0]}\n\n### Negative Part\n{process_MCQ_caption(s)[1]}")
    mcq_templated_df['templated_caption'] = mcq_templated_df.apply(lambda row: [row['caption_0'], row['caption_1'], row['caption_2'], row['caption_3']], axis=1)
    mcq_templated_caps = mcq_templated_df['templated_caption'].explode(ignore_index=True)
    mcq_dataset = pd.concat([mcq_repharased_caps, mcq_templated_caps], axis=1)

    retreival_repharased_df = pd.read_csv(args.repharased_Retreival_csv_path)
    retreival_repharased_df['repharased_caption'] = retreival_repharased_df['captions'].apply(lambda caps: eval(caps))
    retreival_repharased_caps = retreival_repharased_df['repharased_caption'].explode(ignore_index=True)


    retreival_templated_df = pd.read_csv(args.templated_Retreival_csv_path)
    retreival_templated_df['templated_caption'] = retreival_templated_df['captions'].apply(lambda caps: eval(caps))
    retreival_templated_caps = retreival_templated_df['templated_caption'].explode(ignore_index=True)
    retreival_templated_caps = retreival_templated_caps.apply(lambda cap: f"### Positive Part\n{preprocess_retreival_caption(cap)[0]}\n\n### Negative Part\n{preprocess_retreival_caption(cap)[1]}")
    retreival_dataset = pd.concat([retreival_repharased_caps, retreival_templated_caps], axis=1)
    

    dataset = pd.concat([mcq_dataset, retreival_dataset], axis=0).reset_index(drop=True)

    dataset = datasets.Dataset.from_pandas(dataset)
    subsets = dataset.train_test_split(train_size=0.8, seed=42, shuffle=True)
    trainset, valset = subsets['train'], subsets['test']

    def preprocess_function(example):
        return {
            "prompt": [{"role": "user", "content": example["repharased_caption"]}],
            "completion": [
                {"role": "assistant", "content": example['templated_caption']}
            ],
        }

    trainset = trainset.map(preprocess_function, remove_columns=trainset.column_names)
    valset = valset.map(preprocess_function, remove_columns=valset.column_names)


    # Load Model

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)

    lora_cfg = LoraConfig(
        r=8, 
        lora_alpha=16, 
        lora_dropout=0.05,
        bias="none", 
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )


    args = SFTConfig(
        output_dir=os.path.join(args.root, "mistral-7B-sft-Negation"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        num_train_epochs=args.num_epochs,
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=args.lr,
        lr_scheduler_type='cosine',
        warmup_ratio=0.1,
        logging_steps=args.logging_steps,
        max_length=args.max_length,
        fp16=False,
        bf16=True,
        optim='paged_adamw_32bit',
        report_to='wandb',
        completion_only_loss=True
    )

    trainer = SFTTrainer(model, args, train_dataset=trainset, eval_dataset=valset, processing_class=tokenizer, peft_config=lora_cfg)
    trainer.train()


if __name__ == "__main__":
    main()