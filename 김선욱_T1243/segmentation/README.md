# P-Stage-3 Segmentation


## Results

Final submission:
|Public LB |Private LB|
|:---|:---|
|0.6674 |0.6366| 
<br/><br/>


## Models

- FPN

- DeepLabV3Plus<br/><br/>



## Augmentation

```python
A.Compose([
            A.OneOf([A.CLAHE(),
            A.IAASharpen(alpha=(0.2, 0.3)),
            A.GaussianBlur(3, p=0.3)]
                            ,p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.15,        contrast_limit=0.2, p=0.5),
            A.Resize(512,512),
            A.HorizontalFlip(),
            A.ShiftScaleRotate(),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0
            ),
            ToTensorV2()
                            ])
```
<br/><br/>

## Loss
<br/>

- SoftCrossEntropyLoss

- DiceLoss

<br/><br/>

## Optimizer


- ReduceLROnPlateau(
