# Lecture 4 FCN의 한계를 극복한 model

- 성능적인 측면 극복
- 속도적인 측면 극복

# 1. FCN의 한계점

## 1.1 객체의 크기가 크거나 작은 경우 예측을 잘 하지 못하는 문제

- 큰 Object ⇒ 지역적인 정보만으로 예측
    - 범퍼 ⇒ 버스로 예측 & 유리창에 비친 자전거 ⇒ 자전거로 예측
- 같은 object여도 다른 labeling
- 작은 Object가 무시되는 문제

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled.png)

## 1.2 Object의 디테일한 모습이 사라지는 문제 발생

- Deconvolution 절차가 간단하여 경계를 학습하기 어려움

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%201.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%201.png)

# 2. Decoder를 개선한 모델

## 2.1 DeconvNet

Decoder를 Encoder와 대칭으로 만든 형태

- Conv ⇒ Deconv & Pooling ⇒ Unpooling
    - Deconv: Transposed Conv + Batch Norm + ReLU

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%202.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%202.png)

Convolution Network는 VGG16 사용

- 13개의 층
- ReLU와 Pooling이 convoution 사이에서 이뤄짐
- 7x7 Conv 및 1x1 Conv 활용

Deconvolution Network는 Unpooling, deconvolution, ReLU등으로 구성

Unpooling vs Deconvolution

- Unpooling: 디테일한 경계 포착
- Transposed Conv: 전반적인 모습을 포착

### 2.1.1 Unpooling

`Pooling`의 경우 `노이즈를 제거`하지만, `정보의 손실`이 발생

- Unpooling은 maxpooling값의 원래 인덱스를 기억
- Unpooling을 통해서 Pooling시에 지워진 경계의 정보를 기록했다가 복원
- 학습이 필요 없기 때문에 속도가 빠름
- sparse한 activation map을 가지기 때문에 이를 채워 줄 필요가 있음
    - 채우는 역할을 Transposed Convolution이 수행

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%203.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%203.png)

Transposed Convoultion

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%204.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%204.png)

### 2.1.2 Deconvolution (Transposed Convolution)

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%205.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%205.png)

`Pooling`의 경우 `노이즈를 제거`하지만, `정보의 손실`이 발생

- Deconvolution layers를 통해서 input object 모양 복원
- 순차적인 층의 구조가 다양한 수준의 모양을 잡아냄
    - 얕은 층: 전반적인 모습(location, shape, region)
    - 깊은 층: 구체적인 모습(복잡한 패턴)

### 2.1.3 Analysis of Deconvolution Network 1

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%206.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%206.png)

### 2.1.4 Analysis of Deconvolution Network 2

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%207.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%207.png)

Deconvolution Network의 Activation map을 보면 층과 pooling 방법에 따라 다른 특징이 있음

- `Unpooling`의 경우 `example-specific`한 구조를 잡아냄 (자세한 구조. c,e,g,i)
- Transposed Conv의 경우 `class-specific`한 구조를 잡아냄 (위의 구조에 빈 부분을 채워넣음, b,d,f,h,j)
    - 실제로 둘을 병행시 FCN보다 DeconvNet의 Activation map이 자세함

PASCAL VOC 2012에서 좋은 성능 달성 (with Augmented Dataset & FCN과 ensnmble) 

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%208.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%208.png)

### 2.1.5 DeconvNet - Code

```python
# Convolution Network
def CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
	return nn.Sequential(
		nn.Conv2d(in_channels=in_channels,
							out_channels=out_channels,
							kernel_size=kernel_size,
							stride=stride,
							padding=padding),
		nn.BatchNorm2d(out_channels), #추가
		nn.ReLU()
	)
```

```python
# Deconvolution Network
def DCB(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
	return nn.Sequential(
		nn.ConvTranspose2d(in_channels=in_channels,
							out_channels=out_channels,
							kernel_size=kernel_size,
							stride=stride,
							padding=padding),
		nn.BatchNorm2d(out_channels), #추가
		nn.ReLU()
)
```

```python
# 224 x 224
# conv1
self.conv1_1 = CBR(3, 64, 3, 1, 1)
self.conv1_2 = CBR(64, 64, 3, 1, 1)
self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True)
'''
return_indices=True
max_pooling에 의해 지워졌던 정보까지 복원하는 것을 담당하는 인자
out과 더불어 indices까지 리턴하여 2개의 값을 리턴한다.
'''

# 112 x 112
# conv2
self.conv2_1 = CBR(64, 128, 3, 1, 1)
self.conv2_2 = CBR(128, 128, 3, 1, 1)
self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True)

# 56 x 56
# conv3
self.conv3_1 = CBR(128, 256, 3, 1, 1)
self.conv3_2 = CBR(256, 256, 3, 1, 1)
self.conv3_2 = CBR(256, 256, 3, 1, 1)
self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True)

# 28 x 28
# conv4
self.conv4_1 = CBR(256, 512, 3, 1, 1)
self.conv4_2 = CBR(512, 512, 3, 1, 1)
self.conv4_2 = CBR(512, 512, 3, 1, 1)
self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True)

# 14 x 14
# conv5
self.conv5_1 = CBR(512, 512, 3, 1, 1)
self.conv5_2 = CBR(512, 512, 3, 1, 1)
self.conv5_2 = CBR(512, 512, 3, 1, 1)
self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True)

# 7 x 7
# fc6
self.fc6 = CBR(512, 4096, 7, 1, 0) #kernel7, stride1, padding0
self.drop6 = nn.Dropout2d(0.5)

# 1 x 1
# fc7
self.fc7 = CBR(4096, 4096, 1, 1, 0)
self.drop7 = nn.Dropout2d(0.5)
```

```python
# 7 x 7
# fc6-deconv
self.fc6_deconv = DCB(4096, 512, 7, 1, 0)

# 14 x 14
# unpool5
self.unpool5 = nn.MaxUnpool2d(2, stride=2)
self.deconv5_1 = DCB(512, 512, 3, 1, 1)
self.deconv5_1 = DCB(512, 512, 3, 1, 1)
self.deconv5_1 = DCB(512, 512, 3, 1, 1)

# 28 x 28
# unpool4
self.unpool4 = nn.MaxUnpool2d(2, stride=2)
self.deconv4_1 = DCB(512, 512, 3, 1, 1)
self.deconv4_2 = DCB(512, 512, 3, 1, 1)
self.deconv4_2 = DCB(512, 256, 3, 1, 1)

# 56 x 56
# unpool3
self.unpool3 = nn.MaxUnpool2d(2, stride=2)
self.deconv3_1 = DCB(256, 256, 3, 1, 1)
self.deconv3_2 = DCB(256, 256, 3, 1, 1)
self.deconv3_2 = DCB(256, 128, 3, 1, 1)

# 112 x 112
# unpool2
self.unpool2 = nn.MaxUnpool2d(2, stride=2)
self.deconv2_1 = DCB(128, 128, 3, 1, 1)
self.deconv2_2 = DCB(128, 64, 3, 1, 1)

# 224 x 224
# unpool1
self.unpool1 = nn.MaxUnpool2d(2, stride=2)
self.deconv1_1 = DCB(64, 64, 3, 1, 1)
self.deconv1_2 = DCB(64, 64, 3, 1, 1)

# Score
self.score_fr = nn.Conv2d(64, num_classes, 1, 1, 0, 1)
```

```python
h = self.conv1_1(x)
h = self.conv1_2(x)
h, pool1_indices = self.pool1(h)

# ... #

h = self.unpool1(h, pool1_indices)
```

## 2.2 SegNet - For Realtime Semantic Segmentation

- `Road Scene Understanding application` 분야
    - class를 빠르고, 정확하게 구분할 수 있어야 한다
    - 평가지표로 Time 사용

### 2.2.1 SegNet Architecture

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%209.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%209.png)

1. `Encoder`: VGG16의 13개 층 사용, `Decoder`: VGG16의 13개 층을 뒤집어서 사용
2. `FC Layer` 모두 제거하여 `파라미터 수` 감소
3. `Decoder` 파트에서 `Transposed convolution` 대신 `Convolution` 사용
4. `Encoder` 파트에서 `pretrained` 네트워크 사용
    - Conv + BN + ReLU
- 출력단 2개의 Deconv + 1x1 Conv ⇒ 1개의 Conv + 3x3 Conv

### 2.2.2 SegNet Code

- Encoder: DeconvNet과 동일
- Decoder: Deconv ⇒ Conv로 대체 (in_ch, out_ch, k, stride, padding 인자 값 모두 동일)
- 출력단

    ```python
    self.unpool1 = nn.MaxUnpool2d(2, stride=2)
    self.deconv1_1 = CBR(64, 64, 3, 1, 1)

    self.score_fr = nn.Conv2d(64, num_classes, 3, 1, 1) # kernal_size, stride, padding
    # 왜 3x3 Conv를 사용했는지는 논문에서 밝히고 있지 않다.
    ```

## 2.3 DeconvNet vs SegNet

### 공통

- Encoder & Decoder Network가 대칭으로 이루어진 구조
- Encoder Network에 VGG16 사용
    - 13의 층
    - Conv + BN + ReLU + Pooling

### 차이점

- Encoder Network - FC layer 유무
    - DeconvNet: `FC layer`로 `7x7 Conv` 및 `1x1 Conv` 사용
    - SegNet: 파라미터 수 감소를 위해 `FC layer` 제거 ⇒ 학습 및 추론 시간 감소
- Decoder Network - 구조 차이
    - DeconvNet: Unpooling + Deconvolution + ReLU
    - SegNet: Unpooling + Convolution + ReLU

# 3. Skip Connection을 적용한 models

- `DenseNet`: Dense block 내 이전의 모든 layer에 대해 Skip Connection 적용
- `ResNet`: 직전 layer에 대해 Skip Connection 적용

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2010.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2010.png)

## 3.1 FC DenseNet

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2011.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2011.png)

3가지의 Skip Connection을 `concat` 을 활용하여 사용

1. Dense Block 내에서 모든 layer의 output을 concat
2. Dense Block의 input과 output을 concat
3. 같은 level층의 Encoder의 output을 Decoder의 input과 concat

## 3.2 Unet

같은 level층의 Encoder의 output을 Decoder의 input과 concat 되는 구조가 `4번` 반복

# 4. Receptive Field를 확장시킨 models

## 4.0 Receptive Field를 증가시키는 방법

`Receptive Field` : 뉴런이 얼마나 많은 영역을 바라보는지 ⇒ 클수록 Segmentation에 유리

## 4.0.1 Conv + Maxpooling + Conv 구조

- 3x3 Conv + 3x3 Conv ⇒ RF값은 `5x5`
- 3x3 Conv + 2x2 Max pooling + 3x3 Conv ⇒ RF값은 `8x8`
- 단, `Resolution` 측면에서는 `low feature resolution` 문제점 발생 (feature map 크기와 연관있는듯)
- Maxpooling 적용하지 않은채 `Conv만으로` RF를 높이려면 `Parameter 수`가 너무 커진다.

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2012.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2012.png)

### 4.0.2 Dilated Convolution

Dilated Conv & downsampling ↓ ⇒ RF ↑ & Parameter ↓ & High Resolution

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2013.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2013.png)

- 동일 사이즈의 3x3 Conv의 RF값이 3x3인데 비해, 중간에 `zero padding`을 부여함으로써 RF값이 5x5를 갖는다.
    - rate: pixel과 pixel사이의 간격

## 4.1 DeepLab v1

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2014.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2014.png)

- 기본적으로 VGG구조를 가져옴
- `3x3` Maxpooling 적용
    - kernel_size 3 & stride 2 & padding 1
        - 기존 2x2 Maxpooling: kernel_size 2 & stride 2 & padding 0
    - 동일하게 2배로 줄여주지만, 더 넣은 RF값을 갖게 한다.
    - conv4,5는 `stride 1` 을 주어 `이미지 사이즈 고정` (총 2**3 = 8배 축소, 기존 2**5=32배 축소)
- Conv: Conv + ReLU 구조
- Dilated Convolution 적용
    - rate만큼 `padding` 부여
- 논문에서는 3x3 AvgPooling도 이용하는 테크닉 사용
- 3x3 Conv rate=12: RF값 향상
- `Bi-linear Interpolation`
    - Transposed Convolution 대신 `Bi-linear Upsampling` 방식 활용
    - 이웃 좌표의 값 & 거리를 활용해 값을 채우는 방식

    ```python
    F.interpolate(h, size=(feature_map_h * self.upsampling, feature_map_w * self.upsampling), mode='bilinear', align_corners=False)
    F.interpolate(h, size=(feature_map_h * self.upsampling, feature_map_w * self.upsampling), mode='bilinear', align_corners=True)
    ```

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2015.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2015.png)

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2016.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2016.png)

## 4.2 DilatedNet

### 4.2.1 DilatedNet (Only Front-End Module)

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2017.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2017.png)

### DeepLab v1과의 차이점

- `2x2` MaxPooling 사용 (기존의 2x2 MaxPool과 동일)
- Con4,5의 MaxPooling 제거: 똑같이 1/8의 Featuremap 구성하지만, MaxPooling으로 인해 정보가 사라지는 것 방지
    - Dilated Conv는 여전히 RF값을 높임
- FC6에서 7x7 Conv rate=4 사용
    - DeepLab: 3x3 Conv rate=12
    - `rate=4`: 를 줄여 중간에 0이 너무 많아 생기던 문제 보완
    - `7x7`: 내부적으로 `49`개의 파라미터수를 갖으므로서 더 효과적으로 Feature 추출토록 함
- Up Sampling을 Deconv인 Transposed Convolution 사용

### 성능

DeepLab에 비해 특징이 덜 사라지는 짐

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2018.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2018.png)

### 4.2.2 DilatedNet (Front + Basic Context module)

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2019.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2019.png)

### Basic Context Module

- 다양한 크기의 rate를 갖는 Dilated Conv를 적용하는 구조
- 다양한 크기의 `object`를 추출하는 특징
- rate: 1 > 1 > 2 > 4 > 8 > 1 > 1
- in_channels & out_channels: num_classes
- kernel_size: 마지막 layer만 1, 나머지 3

Upsampling ⇒ 8배 축소된 이미지 ⇒ 8배 확대 시켜 입력과 동일한 사이즈 이미지로 변환

### 성능

좀 더 잘 못 분류되는 경우 감소

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2020.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2020.png)

## 4.3 DeepLab v2

FC layer 부분에 Branch 구조로 변화

- ASPP: `Atrous Spatial Pyramid Pooing`
- Multi-Scale을 가지는 Object 추출 성능 향상

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2021.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2021.png)

## 4.4 PSPNet

### 4.4.1 도입배경 1

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2022.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2022.png)

1. `Mismatched Relationship`
    - e.g. 호수 주변의 보트를 FCN은 차로 예측
    - 이유: 외관이 서로 비슷
    - 아이디어: `주변의 특징`을 고려 (e.g. 물 위의 배)
2. `Confusion Categories`
    - FCN은 고층빌딩와 단순 빌딩을 혼돈하여 예측
    - 원인: ADE20K data set 특성상 비슷한 범주인 빌딩과 고층빌딩 존재
    - 아이디어: `categody간의 관계`를 사용하여 해결 (`global contextual information` 사용)
3. `Inconspicuous Classes`
    - FCN은 베개를 침대보로 예측
    - 원인: 베개의의 객체 사이즈가 작고, 침대보와 같은 무늬로 인한 예측 한계
    - 아이디어: `작은 객체`들도 `global contextual information` 사용

### 4.4.2 도입배경 2

FCN도 Maxpool에 의해 줄였는데 why not?

- Object detectors emerge in deep scene cnns 논문 ⇒ 이론적인 RF와 실제적인 RF 차이가 원인
- 즉 원하는 만큼의 RF를 가지지 못해서 생기는 현상

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2023.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2023.png)

### 4.4.3 Pyramid Pooling Module

`Global average pooling` 도입 ⇒ `Pyramid P ooling Module`

- `Feature Map`에 적용 시켜 `sub-region`을 생성
- `sub-region` 각각에 `Conv` 적용하여 `channel = 1`인 `feature map` 생성
    - 1x1x1, 2x2x1, 3x3x1, 6x6x1 ⇒ like Deeplab v2
- 크기를 맞춰 주기 위한 Upsampling 후 skip connection으로 넘어온 `feature map`과 `concat`

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2024.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2024.png)

### Global Average Pooling

- 주변 정보(문맥)을 파악해서 객체를 예측하는데 사용

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2025.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2025.png)

### Convolution VS Gobal Average Pooling

- Convolution: 어떤 모양을 가지는지 추출
- Global Average Pooling: 전체에서 해당 영역이 차지하는 비율

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2026.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2026.png)

### PSPNet 성능

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2027.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2027.png)

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2028.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2028.png)

## 4.5 DeepLab v3

Global Average Pooling 추가

### v2 ⇒ v3 변경사항

- ASPP에 Global Average Pooling 추가
- 기존 ASPP를 Conv 및 Score 거쳐 sum
    - ⇒ 각각을 `Concat` 후 Score와 sum

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2029.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2029.png)

## 4.6 DeepLab v1 ~ v3 정리

FCN

- 크고 작은 객체 예측 X & Decoder 구조의 단순함의 한계

Receptive Field 차원 극복

- Deeplab v1
    - `Dilated Conv`
- Dilated Conv
    - `Maxpooling` 제거
    - 기존 3x3 rate:12 Conv ⇒ 7x7 rate:4 Conv
- DeepLab v2
    - `ASPP` : `다양한 크기의 Dilated Conv`를 `Branch` 형태로 적용 후 `결합`
- PSPNet
    - 이론과 실제 Receptive Field차이를 극복 위해 `Global Average Pooling` 도입
- DeepLab v3
    - `Global Average Pooling` 에 기존 `ASPP` 를 결합한 형태

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2030.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2030.png)

![Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2031.png](Lecture%204%20FCN%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A8%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%80%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A1%E1%86%AB%20model%209cbd1ae35a4d462c8194af71e2a53f8c/Untitled%2031.png)

# 5. 결론

## 5.1 정리

### 문제점들

1. 객체의 크기가 크거나 작은 경우 예측의 어려움
2. Object의 디테일한 모습이 사라지는 문제
    - 간단한 Decov 구조가 문제 ⇒ Encoder와 대칭 되는 Decoder 구조
    - Unpooling을 통해 경계 부분의 정보 포착
        - Unpooling은 `exmaple-specific`한 구조를 잡아냄 (자세한 구조)
        - Transposed Conv는 `class-specific`한 구조를 잡아냄 (위의 구조에 빈 부분 채워넣음)
            - 둘을 병행시 FCN보다 DeconvNet의 활성화 맵이 디테일함
    - SegNet은 속도를 위해 FC 제거 & 3x3 Conv (Score) & Deconv⇒Conv
- By Skip Connection
    - FC DenseNet : Dense Block 활용
    - Unet: Encoder & Decoder 구조 & Feature간 결합을 하는 Skip-connection 적용
- By Receptive Field
    - 기존의 Conv + Max Pooling + Conv 의 low resolution 한계 ⇒ `Dilated Conv`
        - Conv 내부에 Zero Padding ⇒ 동일한 w 개수로 RF 증가
    - Deeplab v1
        - `Dilated Conv` 첫 적용 (3x3 rate=12)
            - Padding과 stride 적절한 적용을 통해 이미지 크기 유지
        - Upsampling by `Bi-linear Intepolation`
    - DilateNet
        - `2x2 maxpool`을 3개만 사용
        - 7x7 rate=4 Dilated Conv 적용
        - Upsampling by `Deconv`
        - `Basic Context Module`
            - `다양한 크기`의 `Dilated Conv` 적용 For `Multi-scale object`
                - rate = 1, 2, 4, 8, 16
            - `잘못 분류`하는 경우  감소
    - Deeplab v2
        - `Multi-scale object`를 위해 `ASPP` module 제안
            - 다양한 크기의 Dialted Conv를 Branch 형태로 적용 후 sum
    - PSPNet
        - 문제점 제시
            - Mismatched Relationship
            - Confusion Categories
            - Inconspicuous Classes
        - 이론적인 RF와 실제적인 RF간의 차이가 문제임을 지적
        - `Global Average Pooling` 도입
            - 다양한 크기의 GAP를 적용해 sub-region 생성후 Upsamling 통해 크기 맞춰준후 Feature map과 Concat
    - Deeplab v3
        - `ASPP` 부분에 `Global Average Pooling` 추가