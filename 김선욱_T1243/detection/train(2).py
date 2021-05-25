from re import M
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pandas as pd
import numpy as np
import random
from utils import mean_average_precision

random_seed = 21
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def collate_fn(batch):
    return tuple(zip(*batch))


from dataset import get_data

train_dataset, val_dataset, test_dataset = get_data()

import wandb
wandb.init(project="FastRCNN")

train_data_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
)
val_data_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
)
test_data_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 11


in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

num_epochs = 36

def train(model,optimizer):
    best_loss = 1000
    loss_hist = Averager()
    best_map = 0
    for epoch in range(num_epochs):
            loss_hist.reset()
            model.train()
            


            for images, targets, image_ids in tqdm(train_data_loader):
                

                # gpu 계산을 위해 image.to(device)
                images = list(image.float().to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # calculate loss
                loss_dict = model(images, targets)

                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()

                loss_hist.send(loss_value)

                # backward
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            print(f"Epoch #{epoch+1} loss: {loss_hist.value}")
            loss = loss_hist.value
            map = validation(model)
            print(f"Epoch #{epoch+1} map: {map}")
            wandb.log({ "loss": loss, "map": map},step = epoch)

            if loss < best_loss:
                torch.save(model.state_dict(), f'faster_rcnn/faster_rcnn_loss.pth')
                best_loss = loss

            if best_map < map:
                torch.save(model.state_dict(), f'faster_rcnn/faster_rcnn.pth')
                best_map = map

def validation(model):
    print("Start validation")
    model.eval()
    with torch.no_grad():
        pred_boxes = []
        true_boxes = []

        for images, targets, image_ids in (val_data_loader):
            images = list(image.float().to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            
            im_id = image_ids


            output = model(images)
            
            # id class score box

            j = 0
            for t in targets:
                for label, boxes in zip(t['labels'],t['boxes']):
                    tmp = []
                    tmp.append(im_id[j][0])
                    tmp.append(label)
                    tmp.append(1)
                    for b in boxes:
                        tmp.append(b)
                    true_boxes.append(tmp)
                j += 1

            i = 0
            for o in output:
                for score,label,boxes in zip(o['scores'],o['labels'],o['boxes']):
                    tmp = []
                    tmp.append(im_id[i][0])
                    tmp.append(label)
                    tmp.append(score)
                    for b in boxes:
                        tmp.append(b)
                    pred_boxes.append(tmp)
                i+=1
            
        
        map = mean_average_precision(pred_boxes, true_boxes)
        return map



# train(model, optimizer)
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)
score_threshold = 0.05
check_point = f'faster_rcnn/faster_rcnn.pth'

model.load_state_dict(torch.load(check_point))
model.eval()
def valid_fn(test_data_loader, model, device):
    outputs = []
    for images, targets, image_ids in tqdm(test_data_loader):
        # gpu 계산을 위해 image.to(device)
        images = list(image.float().to(device) for image in images)
        output = model(images)
        for out in output:
            outputs.append({'boxes': out['boxes'].tolist(), 'scores': out['scores'].tolist(), 'labels': out['labels'].tolist()})
    return outputs

outputs = valid_fn(test_data_loader, model, device)
prediction_strings = []
file_names = []
annotation_test = '../input/data/test.json'
coco = COCO(annotation_test)

score_threshold_label = [0.01, 0.05, 0.075, 0.05, 0.05, 0.05, 0.05, 0.05, 0.075, 0.01, 0.01]

for i, output in enumerate(outputs):
    prediction_string = ''
    image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
    for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
        if score > score_threshold:
            prediction_string += str(label) + ' ' + str(score) + ' ' + str(box[0]) + ' ' + str(
                box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' '
    prediction_strings.append(prediction_string)
    file_names.append(image_info['file_name'])
submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names
submission.to_csv(f'faster_rcnn_submission.csv', index=None)