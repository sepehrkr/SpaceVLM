import clip
import torch
import numpy as np
import pandas as pd
import ast
from transformers import CLIPTokenizer
import math
from torch.utils.data import Dataset, DataLoader, default_collate
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torchvision import transforms
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import seaborn as sns
import os
import argparse


class MCQDataset(Dataset):
    def __init__(self, file_path, transforms):
        super(MCQDataset, self).__init__()
        
        self.df = pd.read_csv(file_path)
        self.transform = transforms
        
        
    def __getitem__(self, index):
        image = self.transform(Image.open(self.df.iloc[index]['image_path']))
        caption_0 = self.df.iloc[index]['caption_0']
        caption_1 = self.df.iloc[index]['caption_1']
        caption_2 = self.df.iloc[index]['caption_2']
        caption_3 = self.df.iloc[index]['caption_3']
        label = self.df.iloc[index]['correct_answer']
        template = self.df.iloc[index]['correct_answer_template']
        
        return image, caption_0, caption_1, caption_2, caption_3, label, template
        
    def __len__(self):
        return len(self.df)


class MCQVideoDataset(Dataset):
    def __init__(self, file_path, transforms):
        super(MCQVideoDataset, self).__init__()
        
        self.df = pd.read_csv(file_path)
        self.transforms = transforms

        self.all_frames = []
        self.create_data()
    
    def create_data(self):
        for video_path in tqdm(self.df['image_path'], total=len(self.df), desc="Extracting frames ..."):
            video_path = "data/msr-vtt/TestVideo/" + video_path
            self.all_frames.append(self.sample_frames(video_path))


    def sample_frames(self, video_path, num_frames=4):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            raise ValueError("Video has no frames or cannot be read.")

        # Compute frame indices to sample
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        sampled_images = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            if not success:
                continue
            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            sampled_images.append(pil_img)

        cap.release()
        return sampled_images


    def check_neg_caption_format(self, caption):
        if caption.startswith('No'):
            caption = caption.split('No')[1].strip()
            
        elif caption == "":
            pass
        
        else:
            raise Exception(f"index {caption}: {caption} -- Format is incorrect!") 
        
        return caption
        
        
    def __getitem__(self, index):
        
        frames = self.all_frames[index]
        transformed_frames = []
        for frame in frames:
            transformed_frames.append(self.transforms(frame))
        
        transformed_frames = torch.stack(transformed_frames, dim=0)


        caption_0 = self.df.iloc[index]['caption_0']
        caption_1 = self.df.iloc[index]['caption_1']
        caption_2 = self.df.iloc[index]['caption_2']
        caption_3 = self.df.iloc[index]['caption_3']
        label = self.df.iloc[index]['correct_answer']
        template = self.df.iloc[index]['correct_answer_template']
        
        return transformed_frames, caption_0, caption_1, caption_2, caption_3, label, template
       
        
    def __len__(self):
        return len(self.df)


@torch.no_grad()
def text_embedding_compute(caption, llm, llm_tokenizer, image_embed, model, device, threshold):

    with torch.no_grad():
        inputs = llm_tokenizer.apply_chat_template(
            [[{'role': 'user', 'content': cap}] for cap in caption],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding='max_length',
            max_length=128,
            truncation=True,
        ).to(llm.device)

        outputs = llm.generate(**inputs, max_new_tokens=128)
        outputs = llm_tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        
        aff_caption = [output.split('### Negative Part')[0].split('### Positive Part')[1].strip() if '### Positive Part' in output else "This is a photo" for output in outputs]
        aff_caption = [text if text else "This is a photo" for text in aff_caption]
        
        neg_caption = ["A photo of a " + output.split('### Negative Part')[1].strip() if '### Negative Part' in output else "" for output in outputs]    


    aff_mask = torch.from_numpy(np.array(aff_caption) != '').to(device)
    neg_mask = torch.from_numpy(np.array(neg_caption) != '').to(device)
    
    only_aff_mask = aff_mask * ~neg_mask
    only_neg_mask = ~aff_mask * neg_mask
    hybrid_mask = aff_mask * neg_mask
    
    assert (only_neg_mask + only_aff_mask + hybrid_mask).all()
    assert ~(only_neg_mask * only_aff_mask * hybrid_mask).all()
    assert (~only_neg_mask).all()
    
    aff_embed = F.normalize(model.encode_text(clip.tokenize(aff_caption).to(device)), dim=-1)
    neg_embed = F.normalize(model.encode_text(clip.tokenize(neg_caption).to(device)), dim=-1)
    
    text_features = torch.zeros(aff_embed.shape, device=aff_embed.device, dtype=aff_embed.dtype)
    # only aff
    text_features[only_aff_mask] = aff_embed[only_aff_mask]
    
    # only neg
    scores = torch.sum(image_embed * neg_embed, dim=-1) # (B,)
    mask_pass = (scores <= threshold) * only_neg_mask
    mask_reject = (scores > threshold) * only_neg_mask
    text_features[mask_pass] = image_embed[mask_pass]
    text_features[mask_reject] = -neg_embed[mask_reject]
    
    # hybrid
    alpha = math.acos(threshold)
    a_T_n = torch.sum(aff_embed * neg_embed, dim=-1) # cos(theta) : B,
    theta = torch.acos(a_T_n) # Theta : (B,)
    mask = theta < 2 * alpha
    mask_pass = (theta < 2 * alpha) * hybrid_mask
    mask_reject = (theta >= 2 * alpha) * hybrid_mask
    delta = alpha + theta/2.
    hybrid_embed = torch.cos(delta)[:, None] * neg_embed + (torch.sin(delta)/torch.sin(theta))[:, None] * (aff_embed - a_T_n[:, None] * neg_embed)
    text_features[mask_pass] = hybrid_embed[mask_pass]   
    text_features[mask_reject] = aff_embed[mask_reject] 
    
    return text_features


@torch.no_grad()
def retreival_evaluation(model, llm, llm_tokenizer, dataLoader, device, threshold=0.9):    
    accs = []
    templates = []
    
    for image, caption_0, caption_1, caption_2, caption_3, label, template in tqdm(dataLoader, total=len(dataLoader)):
                    
        image = image.to(device)
        label = label.to(device)

        # Compute image embeddings
        image_embed = F.normalize(model.encode_image(image), dim=-1) # B x d
        
        # Compute text embeddings
        caption_0_embed = text_embedding_compute(caption_0, llm, llm_tokenizer, image_embed, model, device, threshold)
        caption_1_embed = text_embedding_compute(caption_1, llm, llm_tokenizer, image_embed, model, device, threshold)
        caption_2_embed = text_embedding_compute(caption_2, llm, llm_tokenizer, image_embed, model, device, threshold)
        caption_3_embed = text_embedding_compute(caption_3, llm, llm_tokenizer, image_embed, model, device, threshold)
        captions_embed = torch.stack([caption_0_embed, caption_1_embed, caption_2_embed, caption_3_embed], dim=-2) # B x 4 x d
        
        # Compute scores
        scores = torch.squeeze(image_embed[:, None, :] @ captions_embed.permute((0,2,1)), dim=-2) # B x 4
        preds = torch.argmax(scores, dim=-1) # (B,)
        accs.append((preds == label).float().cpu().numpy())
        templates.extend(template)
        
    accs = np.concatenate(accs, axis=0)
    templates = np.array(templates)

    accs_per_template = {}
    
    for acc, template in zip(accs, templates):
        if template not in accs_per_template:
            accs_per_template[template] = []
            
        accs_per_template[template].append(acc)
    
    accs_per_template = {k: np.mean(v) for k, v in accs_per_template.items()}
    print(" Accuracies: ", accs_per_template)
    print(" Mean Accuracy: ", np.mean(accs))


def video_retreival_evaluation(model, llm, llm_tokenizer, dataLoader, device, threshold=0.9):    
    accs = []
    templates = []
    
    for image, caption_0, caption_1, caption_2, caption_3, label, template in tqdm(dataLoader, total=len(dataLoader)):
                    
        image = image.to(device)
        label = label.to(device)

        # Compute image embeddings
        image_embed = F.normalize(model.encode_image(image.reshape((-1, *image.shape[2:]))), dim=-1)
        image_embed = image_embed.reshape((image.shape[0], image.shape[1], image_embed.shape[-1]))
        image_embed = image_embed.mean(dim=1)
        
        # Compute text embeddings
        caption_0_embed = text_embedding_compute(caption_0, llm, llm_tokenizer, image_embed, model, device, threshold)
        caption_1_embed = text_embedding_compute(caption_1, llm, llm_tokenizer, image_embed, model, device, threshold)
        caption_2_embed = text_embedding_compute(caption_2, llm, llm_tokenizer, image_embed, model, device, threshold)
        caption_3_embed = text_embedding_compute(caption_3, llm, llm_tokenizer, image_embed, model, device, threshold)
        captions_embed = torch.stack([caption_0_embed, caption_1_embed, caption_2_embed, caption_3_embed], dim=-2) # B x 4 x d
        
        # Compute scores
        scores = torch.squeeze(image_embed[:, None, :] @ captions_embed.permute((0,2,1)), dim=-2) # B x 4
        preds = torch.argmax(scores, dim=-1) # (B,)
        accs.append((preds == label).float().cpu().numpy())
        templates.extend(template)
        
    accs = np.concatenate(accs, axis=0)
    templates = np.array(templates)

    accs_per_template = {}
    
    for acc, template in zip(accs, templates):
        if template not in accs_per_template:
            accs_per_template[template] = []
            
        accs_per_template[template].append(acc)
    
    accs_per_template = {k: np.mean(v) for k, v in accs_per_template.items()}
    print(" Accuracies: ", accs_per_template)
    print(" Mean Accuracy: ", np.mean(accs))


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--project_root", type=str, default='./Negation')
    parser.add_argument("--dataset", type=str, default='coco', choices=['coco', 'voc2007', 'msr_vtt'])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.92, help="Usually, best threhsold is determined by cross validating in range [0.9, 0.95]")
    parser.add_argument("--pretrained_model", type=str, default='clip', choices=['clip', 'clip_negfull', 'con_clip', 'neg_clip', 'neg_clip_negfull'])
    parser.add_argument("--clip_backbone", type=str, default='ViT-B/32', choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"])
    parser.add_argument("--llm_finetuned_path", type=str, default='mistral-7B-sft-Negation/checkpoint-426')
    
    args = parser.parse_args()

    os.chdir(args.project_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(args.clip_backbone)
    preprocess = transforms.Compose([transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC, antialias=True)] + preprocess.transforms)
    
    if args.pretrained_model == 'clip_negfull':
        model.load_state_dict(torch.load("models/CLIP_CC12M_NegFull/checkpoint.pt", weights_only=False)['state_dict'])
    
    elif args.pretrained_model == 'con_clip':
        model.load_state_dict(torch.load("models/ConCLIP/conclip_b32_openlip_version.pt"))
        
    elif args.pretrained_model == 'neg_clip':
        model.load_state_dict(torch.load("models/NegCLIP/negclip.pth", weights_only=False)['state_dict'])
        
    elif args.pretrained_model == 'neg_clip_negfull':
        model.load_state_dict(torch.load("models/NegCLIP_CC12M_NegFull/checkpoint.pt", weights_only=False)['state_dict'])
    
        
    model = model.to(device)
    model.eval()
    model = model.to(torch.float32)

    for param in model.parameters():
        param.requires_grad = False


    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_finetuned_path)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token


    llm = AutoModelForCausalLM.from_pretrained(
                        args.llm_finetuned_path,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        attn_implementation="sdpa"
        )
        

    if args.dataset == 'coco':
        dataset = MCQDataset(file_path='data/images/COCO_val_mcq_llama3.1_rephrased.csv', transforms=preprocess)
        dataLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        print("Threshold: ", args.threshold, ", Pretrained Model: ", args.pretrained_model, ", Backbone: ", args.clip_backbone, ", Dataset: ", args.dataset)
        retreival_evaluation(model, llm, llm_tokenizer, dataLoader, device, args.threshold)
        
    elif args.dataset == 'voc2007':
        dataset = MCQDataset(file_path='Negation/data/images/VOC2007_mcq_llama3.1_rephrased.csv', transforms=preprocess)
        dataLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        print("Threshold: ", args.threshold, ", Pretrained Model: ", args.pretrained_model, ", Backbone: ", args.clip_backbone, ", Dataset: ", args.dataset)
        retreival_evaluation(model, llm, llm_tokenizer, dataLoader, device, args.threshold)
        
    elif args.dataset == 'msr_vtt':
        dataset = MCQVideoDataset(file_path='data/videos/msr_vtt_mcq_rephrased_llama.csv', transforms=preprocess)
        dataLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        print("Threshold: ", args.threshold, ", Pretrained Model: ", args.pretrained_model, ", Backbone: ", args.clip_backbone, ", Dataset: ", args.dataset)
        video_retreival_evaluation(model, llm, llm_tokenizer, dataLoader, device, args.threshold)


if __name__ == "__main__":
    main()
