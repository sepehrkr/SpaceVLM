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
from torchvision import transforms
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
import torch.nn as nn
import seaborn as sns
import cv2
import os


THRESHOLDS = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90]

class MCQDataset(Dataset):
    def __init__(self, file_path, transforms):
        super(MCQDataset, self).__init__()
        
        self.df = pd.read_csv(file_path)
        self.df['caption_0'] = self.df['caption_0'].apply(lambda x: eval(x))
        self.df['caption_1'] = self.df['caption_1'].apply(lambda x: eval(x))
        self.df['caption_2'] = self.df['caption_2'].apply(lambda x: eval(x))
        self.df['caption_3'] = self.df['caption_3'].apply(lambda x: eval(x))
        self.transform = transforms
    
    def check_neg_caption_format(self, caption):
        if caption.startswith('No'):
            caption = caption.split('No')[1].strip()
            
        elif caption == "":
            pass
        
        else:
            raise Exception(f"index {caption}: {caption} -- Format is incorrect!") 
        
        return caption
        
        
    def __getitem__(self, index):
        image = self.transform(Image.open(self.df.iloc[index]['image_path']))
        caption_0_pos, caption_0_neg = self.df.iloc[index]['caption_0']
        caption_1_pos, caption_1_neg = self.df.iloc[index]['caption_1']
        caption_2_pos, caption_2_neg = self.df.iloc[index]['caption_2']
        caption_3_pos, caption_3_neg = self.df.iloc[index]['caption_3']
        
        caption_0_pos = "A photo of a " + caption_0_pos if caption_0_pos != "" else "This is a photo"
        caption_1_pos = "A photo of a " + caption_1_pos if caption_1_pos != "" else "This is a photo"
        caption_2_pos = "A photo of a " + caption_2_pos if caption_2_pos != "" else "This is a photo"
        caption_3_pos = "A photo of a " + caption_3_pos if caption_3_pos != "" else "This is a photo"
            
        caption_0_neg = "A photo of a " + caption_0_neg if caption_0_neg != "" else caption_0_neg
        caption_1_neg = "A photo of a " + caption_1_neg if caption_1_neg != "" else caption_1_neg
        caption_2_neg = "A photo of a " + caption_2_neg if caption_2_neg != "" else caption_2_neg
        caption_3_neg = "A photo of a " + caption_3_neg if caption_3_neg != "" else caption_3_neg
        
        
        label = self.df.iloc[index]['correct_answer']
        template = self.df.iloc[index]['correct_answer_template']
        
        return image, caption_0_pos, caption_0_neg, caption_1_pos, caption_1_neg, caption_2_pos, caption_2_neg,\
                caption_3_pos, caption_3_neg, label, template
        
    def __len__(self):
        return len(self.df)


class MCQVideoDataset(Dataset):
    def __init__(self, file_path, transforms):
        super(MCQVideoDataset, self).__init__()
        
        self.df = pd.read_csv(file_path)
        self.df['caption_0'] = self.df['caption_0'].apply(lambda x: eval(x))
        self.df['caption_1'] = self.df['caption_1'].apply(lambda x: eval(x))
        self.df['caption_2'] = self.df['caption_2'].apply(lambda x: eval(x))
        self.df['caption_3'] = self.df['caption_3'].apply(lambda x: eval(x))
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


        caption_0_pos, caption_0_neg = self.df.iloc[index]['caption_0']
        caption_1_pos, caption_1_neg = self.df.iloc[index]['caption_1']
        caption_2_pos, caption_2_neg = self.df.iloc[index]['caption_2']
        caption_3_pos, caption_3_neg = self.df.iloc[index]['caption_3']
        
        caption_0_pos = "A photo of a " + caption_0_pos if caption_0_pos != "" else "This is a photo"
        caption_1_pos = "A photo of a " + caption_1_pos if caption_1_pos != "" else "This is a photo"
        caption_2_pos = "A photo of a " + caption_2_pos if caption_2_pos != "" else "This is a photo"
        caption_3_pos = "A photo of a " + caption_3_pos if caption_3_pos != "" else "This is a photo"
            
        caption_0_neg = "A photo of a " + caption_0_neg if caption_0_neg != "" else caption_0_neg
        caption_1_neg = "A photo of a " + caption_1_neg if caption_1_neg != "" else caption_1_neg
        caption_2_neg = "A photo of a " + caption_2_neg if caption_2_neg != "" else caption_2_neg
        caption_3_neg = "A photo of a " + caption_3_neg if caption_3_neg != "" else caption_3_neg
        
        
        label = self.df.iloc[index]['correct_answer']
        template = self.df.iloc[index]['correct_answer_template']
        
        return transformed_frames, caption_0_pos, caption_0_neg, caption_1_pos, caption_1_neg, caption_2_pos, caption_2_neg,\
                caption_3_pos, caption_3_neg, label, template
        
    def __len__(self):
        return len(self.df)

@torch.no_grad()
def text_embedding_compute(aff_caption, neg_caption, image_embed, model, device, threshold):
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
def retreival_evaluation(model, dataLoader, device, threshold=0.9):    
    accs = []
    templates = []
    
    for image, caption_0_pos, caption_0_neg, caption_1_pos, caption_1_neg, caption_2_pos, caption_2_neg,\
                caption_3_pos, caption_3_neg, label, template in tqdm(dataLoader, total=len(dataLoader)):
                    
        image = image.to(device)
        label = label.to(device)

        # Compute image embeddings
        image_embed = F.normalize(model.encode_image(image), dim=-1) # B x d
        
        # Compute text embeddings
        caption_0_embed = text_embedding_compute(caption_0_pos, caption_0_neg, image_embed, model, device, threshold)
        caption_1_embed = text_embedding_compute(caption_1_pos, caption_1_neg, image_embed, model, device, threshold)
        caption_2_embed = text_embedding_compute(caption_2_pos, caption_2_neg, image_embed, model, device, threshold)
        caption_3_embed = text_embedding_compute(caption_3_pos, caption_3_neg, image_embed, model, device, threshold)
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

def video_retreival_evaluation(model, dataLoader, device, threshold=0.9):    
    accs = []
    templates = []
    
    for image, caption_0_pos, caption_0_neg, caption_1_pos, caption_1_neg, caption_2_pos, caption_2_neg,\
                caption_3_pos, caption_3_neg, label, template in tqdm(dataLoader, total=len(dataLoader)):
                    
        image = image.to(device)
        label = label.to(device)

        # Compute image embeddings
        image_embed = F.normalize(model.encode_image(image.reshape((-1, *image.shape[2:]))), dim=-1)
        image_embed = image_embed.reshape((image.shape[0], image.shape[1], image_embed.shape[-1]))
        image_embed = image_embed.mean(dim=1)
        
        # Compute text embeddings
        caption_0_embed = text_embedding_compute(caption_0_pos, caption_0_neg, image_embed, model, device, threshold)
        caption_1_embed = text_embedding_compute(caption_1_pos, caption_1_neg, image_embed, model, device, threshold)
        caption_2_embed = text_embedding_compute(caption_2_pos, caption_2_neg, image_embed, model, device, threshold)
        caption_3_embed = text_embedding_compute(caption_3_pos, caption_3_neg, image_embed, model, device, threshold)
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





os.chdir('/home/amirhossein.hajimohammadrezaie/ECOR_extended/Negation')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device=device)
preprocess = transforms.Compose([transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC, antialias=True)] + preprocess.transforms)
model.eval()
model = model.to(torch.float32)
batch_size = 128

for param in model.parameters():
    param.requires_grad = False

#dataset = MCQDataset(file_path='data/images/VOC2007_mcq_splitted.csv', transforms=preprocess)   
dataset = MCQVideoDataset(file_path='data/videos/msr_vtt_mcq_splitted.csv', transforms=preprocess)   


for threshold in THRESHOLDS:
    print("Threshold: ", threshold)
    dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8) 
    #retreival_evaluation(model, dataLoader, device, threshold)
    video_retreival_evaluation(model, dataLoader, device, threshold)

