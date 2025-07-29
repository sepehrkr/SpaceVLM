import clip
import torch
import numpy as np
import pandas as pd
import ast
from transformers import CLIPTokenizer
import math
from torch.utils.data import Dataset, DataLoader, default_collate
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
import torch.nn as nn
import seaborn as sns
import os


THRESHOLDS = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90]

class CsvImageCaptionDataset(Dataset):
    def __init__(self, csv_file, transforms, sep=',', img_key='filepath'):
        
        self.df = pd.read_csv(csv_file, sep=sep)
        self.df['negated_captions'] = self.df.apply(lambda row: [f"a photo of a {neg_obj}" for cap in eval(row['captions']) for neg_obj in eval(row['negative_objects'])], axis=1)
        self.df['captions'] = self.df.apply(lambda row: [cap for cap in eval(row['captions']) for neg_obj in eval(row['negative_objects'])], axis=1)

        self.transforms = transforms
        self.img_key = img_key

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx][self.img_key]
        image = self.transforms(Image.open(image_path))
        captions = self.df.iloc[idx]['captions']
        neg_captions = self.df.iloc[idx]['negated_captions']
        return image, captions, neg_captions

class CsvVideoCaptionDataset(Dataset):
    def __init__(self, csv_file, transforms, sep=',', img_key='image_id'):
        
        self.df = pd.read_csv(csv_file, sep=sep)
        
        self.df['captions'] = self.df['captions'].apply(lambda caps: eval(caps))
        self.df['negated_captions'] = self.df.apply(lambda row: [f"a photo of a {row['negative_concept']}"], axis=1)

        self.transforms = transforms
        self.img_key = img_key

        self.all_frames = []
        self.create_data()


    def create_data(self):
        for video_path in tqdm(self.df[self.img_key], total=len(self.df), desc="Extracting frames ..."):
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
        captions = self.df.iloc[idx]['captions']
        neg_captions = self.df.iloc[idx]['negated_captions']
        return transformed_frames, captions, neg_captions


def image_captions_collate_fn(batch):
    images, texts1, texts2 = list(zip(*batch))
    images = default_collate(images)
    return images, texts1, texts2

def dataloader_with_indices(dataloader):
    start = 0
    for x, y1, y2 in dataloader:
        end = start + len(x)
        inds = torch.arange(start, end)
        yield x, y1, y2, inds
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
def retreival_evaluation(model, dataLoader, device, threshold=0.9):    
    batch_images_emb_list = []
    batch_texts_emb_list = []
    texts_image_index = []
    
    for batch_images, batch_aff_texts, batch_not_texts, inds in tqdm(dataLoader, total=len_dataLoader):
        batch_images = batch_images.to(device)
        batch_aff_texts_tok = clip.tokenize([text for i, texts in enumerate(batch_aff_texts) for text in texts], truncate=True).to(device)
        batch_not_texts_tok = clip.tokenize([text for i, texts in enumerate(batch_not_texts) for text in texts], truncate=True).to(device)
        batch_texts_image_index = [ind for ind, texts in zip(inds, batch_aff_texts) for text in texts]

        # Compute image embeddings
        batch_images_emb = F.normalize(model.encode_image(batch_images), dim=-1)
        
        # Compute text embeddings
        batch_aff_texts_emb = F.normalize(model.encode_text(batch_aff_texts_tok), dim=-1) # a : B x d
        batch_not_texts_emb = F.normalize(model.encode_text(batch_not_texts_tok), dim=-1) # n : B x d
        alpha = math.acos(threshold) 
        a_T_n = torch.sum(batch_aff_texts_emb * batch_not_texts_emb, dim=-1) # cos(theta) : B,
        theta = torch.acos(a_T_n) # Theta : B,
        mask = theta < 2 * alpha

        delta = alpha + theta/2.
        batch_texts_emb = torch.cos(delta)[:, None] * batch_not_texts_emb + (torch.sin(delta)/torch.sin(theta))[:, None] * (batch_aff_texts_emb - a_T_n[:, None]*batch_not_texts_emb)
        
        #batch_texts_emb = torch.zeros_like(batch_neg_texts_emb1)
        #batch_texts_emb[mask] = batch_neg_texts_emb1[mask]
        #batch_texts_emb[~mask] = batch_aff_texts_emb[~mask]
        
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
    img_retrieval_5 = (batchify(recall_at_k, scores, positive_pairs, samples_num, device, k=5)>0).float().mean().item()
    txt_retrieval_5= (batchify(recall_at_k, scores.T, positive_pairs.T, samples_num, device, k=5)>0).float().mean().item()

    print(f"Image retrieval at 5: {img_retrieval_5:.3g}")
    print(f"Text retrieval at 5: {txt_retrieval_5:.3g}")


@torch.no_grad()
def video_retreival_evaluation(model, dataLoader, device, threshold=0.9):    
    batch_images_emb_list = []
    batch_texts_emb_list = []
    texts_image_index = []
    
    for batch_images, batch_aff_texts, batch_not_texts, inds in tqdm(dataLoader, total=len_dataLoader):
        batch_images = batch_images.to(device)
        batch_aff_texts_tok = clip.tokenize([text for i, texts in enumerate(batch_aff_texts) for text in texts], truncate=True).to(device)
        batch_not_texts_tok = clip.tokenize([text for i, texts in enumerate(batch_not_texts) for text in texts], truncate=True).to(device)
        batch_texts_image_index = [ind for ind, texts in zip(inds, batch_aff_texts) for text in texts]

        # Compute image embeddings
        batch_images_emb = F.normalize(model.encode_image(batch_images.reshape((-1, *batch_images.shape[2:]))), dim=-1)
        batch_images_emb = batch_images_emb.reshape((batch_images.shape[0], batch_images.shape[1], batch_images_emb.shape[-1]))
        batch_images_emb = batch_images_emb.mean(dim=1)
        
        # Compute text embeddings
        batch_aff_texts_emb = F.normalize(model.encode_text(batch_aff_texts_tok), dim=-1) # a : B x d
        batch_not_texts_emb = F.normalize(model.encode_text(batch_not_texts_tok), dim=-1) # n : B x d
        alpha = math.acos(threshold) 
        a_T_n = torch.sum(batch_aff_texts_emb * batch_not_texts_emb, dim=-1) # cos(theta) : B,
        theta = torch.acos(a_T_n) # Theta : B,
        mask = theta < 2 * alpha

        delta = alpha + theta/2.
        batch_neg_texts_emb = torch.cos(delta)[:, None] * batch_not_texts_emb + (torch.sin(delta)/torch.sin(theta))[:, None] * (batch_aff_texts_emb - a_T_n[:, None]*batch_not_texts_emb)
        
        batch_texts_emb = torch.zeros_like(batch_neg_texts_emb)
        batch_texts_emb[mask] = batch_neg_texts_emb[mask]
        batch_texts_emb[~mask] = batch_aff_texts_emb[~mask]
        
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
    img_retrieval_5 = (batchify(recall_at_k, scores, positive_pairs, samples_num, device, k=5)>0).float().mean().item()
    txt_retrieval_5= (batchify(recall_at_k, scores.T, positive_pairs.T, samples_num, device, k=5)>0).float().mean().item()

    print(f"Image retrieval at 5: {img_retrieval_5:.3g}")
    print(f"Text retrieval at 5: {txt_retrieval_5:.3g}")


os.chdir('/home/amirhossein.hajimohammadrezaie/ECOR_extended/Negation')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device=device)
preprocess = transforms.Compose([transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC, antialias=True)] + preprocess.transforms)
model.eval()
model = model.to(torch.float32)
batch_size = 128

for param in model.parameters():
    param.requires_grad = False

dataset = CsvImageCaptionDataset(csv_file='data/images/COCO_val_retrieval.csv', transforms=preprocess, img_key='filepath')
#dataset = CsvVideoCaptionDataset(csv_file='data/videos/msr_vtt_retrieval_neg.csv', transforms=preprocess, img_key='image_id')

for threshold in THRESHOLDS:
    dataLoader = DataLoader(dataset, batch_size, collate_fn=image_captions_collate_fn, shuffle=False, num_workers=16)
    len_dataLoader = len(dataLoader)
    dataLoader = dataloader_with_indices(dataLoader)
    print("Threshold: ", threshold)
    retreival_evaluation(model, dataLoader, device, threshold)
    #video_retreival_evaluation(model, dataLoader, device, threshold)

