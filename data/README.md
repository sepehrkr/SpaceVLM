# Data Directory

This directory contains the necessary `.csv` files for the experiments. To fully use the repository, you need to download the corresponding image and video datasets listed below.

## 📥 Required Datasets

- **COCO 2017 Validation Images:** [Download here](https://cocodataset.org/#download)  
- **VOC2007:** [Download here](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)  
- **MSR-VTT Test Split:** [Download here](https://www.kaggle.com/datasets/vishnutheepb/msrvtt)  

> Place the downloaded files in the corresponding folders as shown in the directory structure below.

## 📂 Directory Structure
'''
  data
  ├── images # Provided
  ├── videos # Provided
  ├── coco
  │ └── images
  │ └── val2017
  │ ├── 000000000139.jpg
  │ ├── 000000000285.jpg
  │ └── ...
  ├── voc2007
  │ └── VOCdevkit
  │ └── VOC2007
  │ └── JPEGImages
  │ ├── 000001.jpg
  │ ├── 000002.jpg
  │ └── ...
  └── msr-vtt
  └── TestVideo
  ├── video7010.mp4
  ├── video7011.mp4
  └── ...
'''
