import clip
import torch
import numpy as np
import pandas as pd
import ast
from transformers import CLIPTokenizer
import math
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import Dataset, DataLoader, default_collate
from tqdm import tqdm
import cv2

import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import seaborn as sns
import os
import argparse


class CsvImageCaptionDataset(Dataset):
    def __init__(self, csv_file, transforms, sep=',', img_key='filepath', caption_key='captions'):
        
        self.df = pd.read_csv(csv_file, sep=sep)
        self.df['captions'] = self.df['captions'].apply(lambda caps: eval(caps))

        self.transforms = transforms
        self.img_key = img_key
        self.caption_key = caption_key

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx][self.img_key]
        images = self.transforms(Image.open(image_path))
        captions = self.df.iloc[idx][self.caption_key]
        return images, captions


class CsvVideoCaptionDataset(Dataset):
    def __init__(self, csv_file, transforms, sep=',', img_key='filepath', caption_key='captions'):
        
        self.df = pd.read_csv(csv_file, sep=sep)
        self.df['captions'] = self.df['captions'].apply(lambda caps: eval(caps))

        self.transforms = transforms
        self.img_key = img_key
        self.caption_key = caption_key

        self.all_frames = []
        self.create_data()


    def create_data(self):
        for video_path in tqdm(self.df['image_id'], total=len(self.df), desc="Extracting frames ..."):
            video_path = "data/msr-vtt/TestVideo/" + video_path + ".mp4"
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        frames = self.all_frames[idx]
        transformed_frames = []
        for frame in frames:
            transformed_frames.append(self.transforms(frame))
        
        transformed_frames = torch.stack(transformed_frames, dim=0)
        captions = self.df.iloc[idx][self.caption_key]
        
        return transformed_frames, captions, len(captions)


def image_captions_collate_fn(batch):
    images, texts = list(zip(*batch))
    images = default_collate(images)
    return images, texts


def dataloader_with_indices(dataloader):
    start = 0
    for x, y in dataloader:
        end = start + len(x)
        inds = torch.arange(start, end)
        yield x, y, inds
        start = end


def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)


def recall_at_k(scores, positive_pairs, k):
    nb_texts, nb_images = scores.shape
    topk_indices = torch.topk(scores, k, dim=1)[1]
    nb_positive = positive_pairs.sum(dim=1)
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1,2))
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k


@torch.no_grad()
def retreival_evaluation(model, dataLoader, len_dataLoader, device):    
    batch_images_emb_list = []
    batch_texts_emb_list = []
    texts_image_index = []

    for batch_images, batch_texts, inds in tqdm(dataLoader, total=len_dataLoader):
        batch_images = batch_images.to(device)
        batch_texts_tok = clip.tokenize([text for i, texts in enumerate(batch_texts) for text in texts], truncate=True).to(device)
        batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts) for text in texts]

        # Compute the embeddings
        batch_images_emb = F.normalize(model.encode_image(batch_images), dim=-1)
        batch_texts_emb = F.normalize(model.encode_text(batch_texts_tok), dim=-1)
        
        # Append the embeddings and indices
        batch_images_emb_list.append(batch_images_emb.cpu())
        batch_texts_emb_list.append(batch_texts_emb.cpu())
        texts_image_index.extend(batch_texts_image_index)


    samples_num = len(batch_images_emb_list[0])
    images_emb = torch.cat(batch_images_emb_list, dim=0)
    texts_emb = torch.cat(batch_texts_emb_list, dim=0)

    # Compute the scores
    scores = texts_emb @ images_emb.t()
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    
    # Compute the recall@k
    img_retrieval_1 = (batchify(recall_at_k, scores, positive_pairs, samples_num, device, k=1)>0).float().mean().item()
    txt_retrieval_1= (batchify(recall_at_k, scores.T, positive_pairs.T, samples_num, device, k=1)>0).float().mean().item()

    print(f"Image retrieval at 1: {img_retrieval_1:.3g}")
    print(f"Text retrieval at 1: {txt_retrieval_1:.3g}")

    img_retrieval_5 = (batchify(recall_at_k, scores, positive_pairs, samples_num, device, k=5)>0).float().mean().item()
    txt_retrieval_5= (batchify(recall_at_k, scores.T, positive_pairs.T, samples_num, device, k=5)>0).float().mean().item()

    print(f"Image retrieval at 5: {img_retrieval_5:.3g}")
    print(f"Text retrieval at 5: {txt_retrieval_5:.3g}")
    
    img_retrieval_10 = (batchify(recall_at_k, scores, positive_pairs, samples_num, device, k=10)>0).float().mean().item()
    txt_retrieval_10 = (batchify(recall_at_k, scores.T, positive_pairs.T, samples_num, device, k=10)>0).float().mean().item()

    print(f"Image retrieval at 10: {img_retrieval_10:.3g}")
    print(f"Text retrieval at 10: {txt_retrieval_10:.3g}")

@torch.no_grad()
def video_retreival_evaluation(model, dataLoader, len_dataLoader, device):    
    batch_images_emb_list = []
    batch_texts_emb_list = []
    texts_image_index = []

    for batch_images, batch_texts, inds in tqdm(dataLoader, total=len_dataLoader):
        batch_images = batch_images.to(device)
        batch_texts_tok = clip.tokenize([text for i, texts in enumerate(batch_texts) for text in texts], truncate=True).to(device)
        batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts) for text in texts]

        # Compute the embeddings        
        batch_images_emb = F.normalize(model.encode_image(batch_images.reshape((-1, *batch_images.shape[2:]))), dim=-1)
        batch_images_emb = batch_images_emb.reshape((batch_images.shape[0], batch_images.shape[1], batch_images_emb.shape[-1]))
        batch_images_emb = batch_images_emb.mean(dim=1)
        
        batch_texts_emb = F.normalize(model.encode_text(batch_texts_tok), dim=-1)
        
        # Append the embeddings and indices
        batch_images_emb_list.append(batch_images_emb.cpu())
        batch_texts_emb_list.append(batch_texts_emb.cpu())
        texts_image_index.extend(batch_texts_image_index)


    samples_num = len(batch_images_emb_list[0])
    images_emb = torch.cat(batch_images_emb_list, dim=0)
    texts_emb = torch.cat(batch_texts_emb_list, dim=0)

    # Compute the scores
    scores = texts_emb @ images_emb.t()
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    
    # Compute the recall@k
    img_retrieval_1 = (batchify(recall_at_k, scores, positive_pairs, samples_num, device, k=1)>0).float().mean().item()
    txt_retrieval_1= (batchify(recall_at_k, scores.T, positive_pairs.T, samples_num, device, k=1)>0).float().mean().item()

    print(f"Image retrieval at 1: {img_retrieval_1:.3g}")
    print(f"Text retrieval at 1: {txt_retrieval_1:.3g}")

    img_retrieval_5 = (batchify(recall_at_k, scores, positive_pairs, samples_num, device, k=5)>0).float().mean().item()
    txt_retrieval_5= (batchify(recall_at_k, scores.T, positive_pairs.T, samples_num, device, k=5)>0).float().mean().item()

    print(f"Image retrieval at 5: {img_retrieval_5:.3g}")
    print(f"Text retrieval at 5: {txt_retrieval_5:.3g}")
    
    img_retrieval_10 = (batchify(recall_at_k, scores, positive_pairs, samples_num, device, k=10)>0).float().mean().item()
    txt_retrieval_10 = (batchify(recall_at_k, scores.T, positive_pairs.T, samples_num, device, k=10)>0).float().mean().item()

    print(f"Image retrieval at 10: {img_retrieval_10:.3g}")
    print(f"Text retrieval at 10: {txt_retrieval_10:.3g}")



def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--project_root", type=str, default='/home/amirhossein.hajimohammadrezaie/ECOR_extended/Negation')
    parser.add_argument("--dataset", type=str, default='coco', choices=['coco', 'msr_vtt'])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--pretrained_model", type=str, default='clip', choices=['clip', 'clip_negfull', 'con_clip', 'neg_clip', 'neg_clip_negfull'])
    parser.add_argument("--clip_backbone", type=str, default='ViT-B/32', choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"])
    parser.add_argument("--neg_retrieval", type=bool, default=False)
    
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
        

    if args.dataset == 'coco':
        if args.neg_retrieval:
            dataset = CsvImageCaptionDataset(csv_file='data/images/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv', transforms=preprocess, img_key='filepath')
        else:
            dataset = CsvImageCaptionDataset(csv_file='data/images/COCO_val_retrieval.csv', transforms=preprocess, img_key='filepath')
        
        dataLoader = DataLoader(dataset, args.batch_size, collate_fn=image_captions_collate_fn, shuffle=False, num_workers=8)
        len_dataLoader = len(dataLoader)
        dataLoader = dataloader_with_indices(dataLoader)
        print("Pretrained Model: ", args.pretrained_model, ", Backbone: ", args.clip_backbone, ", Dataset: ", args.dataset)
        retreival_evaluation(model, dataLoader, len_dataLoader, device)
        
    elif args.dataset == 'msr_vtt':
        if args.neg_retrieval:
            dataset = CsvVideoCaptionDataset(csv_file='data/videos/msr_vtt_retrieval_rephrased_llama.csv', transforms=preprocess, img_key='image_id')
        else:
             dataset = CsvVideoCaptionDataset(csv_file='data/videos/msr_vtt_retrieval.csv', transforms=preprocess, img_key='image_id')
        
        dataLoader = DataLoader(dataset, args.batch_size, collate_fn=image_captions_collate_fn, shuffle=False, num_workers=8)
        len_dataLoader = len(dataLoader)
        dataLoader = dataloader_with_indices(dataLoader)
        print("Pretrained Model: ", args.pretrained_model, ", Backbone: ", args.clip_backbone, ", Dataset: ", args.dataset)
        video_retreival_evaluation(model, dataLoader, len_dataLoader, device)
    

if __name__ == "__main__":
    main()
