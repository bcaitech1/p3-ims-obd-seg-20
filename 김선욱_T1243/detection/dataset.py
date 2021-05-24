from albumentations.augmentations.transforms import ShiftScaleRotate
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from pycocotools.coco import COCO
import cv2
import os
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import pandas as pd
from pycocotools.cocoeval import COCOeval
import random

random_seed = 21
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

class detectionDataset(Dataset):
    def __init__(self,data_dir,annotation,transform):
        super(detectionDataset,self).__init__()
        self.data_dir = data_dir
        self.coco = COCO(annotation)
        self.prediction = {
            "images":self.coco.dataset["images"].copy(),
            "categories":self.coco.dataset["categories"].copy(),
            "annotation":None
        }
        self.transform = transform

    def __getitem__(self, index: int):
        image_id = self.coco.getImgIds(imgIds=index) # images의 id를 뽑음
        images_info = self.coco.load_imgs(image_id)[0] # {[]} 형태 
        image = cv2.imread(os.path.join(self.data_dir,images_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        

        ann_ids = self.coco.getAnnIds(imgIds=images_info['id'])
        anns = self.coco.loadAnns(ann_ids)

        boxes = np.array([x['bbox'] for x in anns])
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        labels = np.array([x['category_id'] for x in anns])
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        areas = np.array([x['area'] for x in anns])
        areas = torch.as_tensor(areas, dtype=torch.float32)
                                
        is_crowds = np.array([x['iscrowd'] for x in anns])
        is_crowds = torch.as_tensor(is_crowds, dtype=torch.int64)
                                
        segmentation = np.array([x['segmentation'] for x in anns], dtype=object)

        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([index]), 'area': areas,
                  'iscrowd': is_crowds}

        # transform
        if self.transform:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transform(**sample)
            image = sample['image']
            target['boxes'] = torch.tensor(sample['bboxes'], dtype=torch.float32)

        return image, target, image_id
    
    def __len__(self) -> int:
        return len(self.coco.getImgIds())


def get_data():
    train_transform = A.Compose([
        
        A.OneOf( [A.RandomRotate90(),
                A.HorizontalFlip()],
                p=1.0),
        A.ShiftScaleRotate(p=0.25),
        
        
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0
        ),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

    val_transform = A.Compose([
        A.Normalize(
                                mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0
                            ),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


    test_transform = A.Compose([
        A.Normalize(
                                mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0
                            ),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


    annotation_train = '../input/data/train.json'
    annotation_val = '../input/data/val.json'
    annotation_test = '../input/data/test.json'
    data_dir = '../input/data'
    train_dataset = detectionDataset(data_dir=data_dir, annotation=annotation_train, transform=train_transform)

    # validation dataset
    val_dataset = detectionDataset(data_dir=data_dir, annotation=annotation_val, transform=val_transform)

    # test dataset
    test_dataset = detectionDataset(data_dir=data_dir, annotation=annotation_test, transform=test_transform)



    return train_dataset, val_dataset, test_dataset