import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import random
import time
import json
import warnings 
warnings.filterwarnings('ignore')


from torch.utils.data import Dataset, DataLoader
from utils import label_accuracy_score
import cv2

import numpy as np
import pandas as pd

# 전처리를 위한 라이브러리
from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp

model_1 = smp.DeepLabV3Plus(
    encoder_name="timm-efficientnet-b3",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="noisy-student",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=12,                      # model output channels (number of classes in your dataset)
)
model_2 = smp.FPN(
    encoder_name="timm-efficientnet-b3",
    encoder_weights="noisy-student",
    in_channels=3,
    classes = 12,
)
model_3 = smp.DeepLabV3Plus(
    encoder_name="se_resnext101_32x4d",
    encoder_weights="imagenet",
    in_channels=3,
    classes = 12,
)


def multi_model_ensemble_test (models, weights, dataloader, device) :
    size = 256
    transform = A.Compose([A.Resize(256, 256)])
    print('Start prediction.')
    for model in models :
        model.eval()    
    file_name_list = []
    #file_list_
    preds_array = np.empty((0, size*size), dtype=np.long)    
    with torch.no_grad():
        for step, (imgs, image_infos) in (enumerate(test_loader)):
            x = torch.stack(imgs).to(device)            # inference (512 x 512)
            outs = models[0](x) * weights[0]
            isFirst = True
            for model, weight in zip(models, weights) :
                if isFirst :
                    isFirst = False
                else :
                    outs += model(x) * weight            
            if outs.shape[0] == 1 : oms = torch.argmax(outs.squeeze(), dim = 0).detach().cpu().numpy()
            else : oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)            
            oms = np.array(temp_mask)            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))            
            file_name_list.append([i['file_name'] for i in image_infos])    
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]    
    return file_names, preds_array

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
models = []
weights = [0.15,0.4,0.45]

model_path1 = './saved/best/DeepLabV3_FINAL.pt'
model_path2 = './saved/best/FPN_FINAL.pt'
model_path3 = './saved/best/DeepLabV3_FINAL2.pth'

checkpoint1 = torch.load(model_path1, map_location=device)
checkpoint2 = torch.load(model_path2, map_location=device)
checkpoint3 = torch.load(model_path3, map_location=device)

model_1.load_state_dict(checkpoint1)
model_1 = model_1.to(device)
models.append(model_1)

model_2.load_state_dict(checkpoint2)
model_2 = model_2.to(device)
models.append(model_2)


model_3.load_state_dict(checkpoint3)
model_3 = model_3.to(device)
models.append(model_3)



# sample_submisson.csv 열기
submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)
# test set에 대한 prediction
file_names, preds = multi_model_ensemble_test(models, weights, test_loader, device)
# PredictionString 대입
for file_name, string in zip(file_names, preds):
    submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                   ignore_index=True)# submission.csv로 저장
save_name = "ensemble"
submission.to_csv("./submission/"+save_name+".csv", index=False)