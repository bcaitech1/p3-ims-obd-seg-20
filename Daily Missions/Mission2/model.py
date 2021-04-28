from collections import OrderedDict
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding




###########################Test
import unittest

from torch.nn.modules.dropout import Dropout2d

class TestModel(unittest.TestCase):

    def setUp(self):
        '''
        테스트가 수행되기 전 환경 설정 
        '''
        self.num_classes,self.height,self.width=[12,512,512]
        self.input=torch.randn(size=[1,3,512,512])

        #model
        self.deeplab_v1=DeepLabV1()
        self.dilated_net=DilatedNet()
        self.deeplab_v2=DeepLabV2()
        print(self.deeplab_v2)

        

    def test_deeplab_v1_shape(self):
        out=self.deeplab_v1(self.input)
        _,out_num_classes,out_height,out_width=out.shape
        self.assertEqual(out_num_classes,self.num_classes,msg='Failed to match "num_classes"')
        self.assertEqual(out_height,self.height,msg='Failed to match "height"')
        self.assertEqual(out_width,self.width,msg='Failed to match "width"')
        print('Success to Deeplab V1')

    def test_deeplab_v2_shape(self):
        out=self.deeplab_v2(self.input)
        _,out_num_classes,out_height,out_width=out.shape
        self.assertEqual(out_num_classes,self.num_classes,msg='Failed to match "num_classes"')
        self.assertEqual(out_height,self.height,msg='Failed to match "height"')
        self.assertEqual(out_width,self.width,msg='Failed to match "width"')
        print('Success to Deeplab V2')
    
    def test_dilated_net_shape(self):
        out=self.dilated_net(self.input)
        _,out_num_classes,out_height,out_width=out.shape
        self.assertEqual(out_num_classes,self.num_classes,msg='Failed to match "num_classes"')
        self.assertEqual(out_height,self.height,msg='Failed to match "height"')
        self.assertEqual(out_width,self.width,msg='Failed to match "width"')
        print('Success to DilatedNet')

        
        





#Backbone
class ResNet(nn.Module):
    """Some Information about ResNet"""
    def __init__(self):
        super(ResNet, self).__init__()

    def forward(self, x):

        return x


'''
=============================================================================
Receptive Field를 확장시킨 models
DeepLabV1,DilatedNet,DeepLabV2,DeepLabV3
=============================================================================
'''

class DeepLabV1(nn.Module):
    """Some Information about DeepLabV1-LargeFOV
    conv1(rate=1) : 3x3 conv , 3x3 conv , 3x3 MaxPooling(stride=2,padding=1)
    conv2(rate=1) : 3x3 conv , 3x3 conv , 3x3 MaxPooling(stride=2,padding=1)
    conv3(rate=1) : 3x3 conv , 3x3 conv , 3x3 conv , 3x3 MaxPooling(stride=2,padding=1)  
    conv4(rate=1) : 3x3 conv , 3x3 conv , 3x3 conv , 3x3 MaxPooling(stride=1,padding=1)
    conv5(rate=2) : 3x3 conv , 3x3 conv , 3x3 conv , 3x3 MaxPooling(stride=1,padding=1) , 3x3 AvgPooling(stride=1,padding=1)
    FC6(rate=12) : 3x3 conv
    FC7 : 1x1 conv
    Score : 1x1 conv
    Up Sampling : Bi-linear Interpolation


    <주목>
    기존의 VGG 의 경우 2x2 MaxPooling(stride=2) 로 1/2 배 줄였지만,
    DeepLabV1에서는 3x3 MaxPooling(stride=2,padding=1) 로 1/2배 줄어드는것은 같지만, receptive field가 더 확장된다. 
    """
    def __init__(self,num_classes=12,upsampling=8):
        super(DeepLabV1, self).__init__()

        def CBR(in_channels,out_channels,kernel_size,stride,rate):
            return nn.Sequential(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=rate, # padding = rate 
                dilation=rate
                ),
                nn.ReLU(inplace=True)

            )
        def vgg16():
            module=nn.ModuleDict()
            #conv1(rate=1)
            conv1=nn.Sequential(
                CBR(in_channels=3,out_channels=64,kernel_size=3,stride=1,rate=1),
                CBR(64,64,3,1,rate=1),
                nn.MaxPool2d(kernel_size=3,stride=2,padding=1) # Original -> 1/2
                
            )
            module['conv1']=conv1

            #conv2(rate=1)
            conv2=nn.Sequential(
                CBR(64,128,3,1,rate=1),
                CBR(128,128,3,1,rate=1),
                nn.MaxPool2d(3,2,1) # Original -> 1/4
            )
            module['conv2']=conv2

            #conv3(rate=1)
            conv3=nn.Sequential(
                CBR(128,256,3,1,rate=1),
                CBR(256,256,3,1,rate=1),
                CBR(256,256,3,1,rate=1),
                nn.MaxPool2d(3,2,1) # Original -> 1/8
            )
            module['conv3']=conv3

            #conv4(rate=1)
            conv4=nn.Sequential(
                CBR(256,512,3,1,rate=1),
                CBR(512,512,3,1,rate=1),
                CBR(512,512,3,1,rate=1),
                nn.MaxPool2d(3,1,1) # stride = 1 -> Fixed image size.  Original -> 1/8(Fixed)
            )
            module['conv4']=conv4

            #conv5(rate=2)
            conv5=nn.Sequential(
                CBR(512,512,3,1,rate=2),
                CBR(512,512,3,1,rate=2),
                CBR(512,512,3,1,rate=2),
                nn.MaxPool2d(3,1,1), # stride = 1 -> Fixed image size. Orginal -> 1/8(Fixed)
                nn.AvgPool2d(3,1,1) # stride = 1 -> Fixed image size. Orginal -> 1/8(Fixed)
            )
            module['conv5']=conv5


            return module

        def classifier():
            module=nn.ModuleDict()
            #fc6(rate=12)
            fc6=nn.Sequential(
                CBR(512,1024,3,1,rate=12),
                nn.Dropout2d()
            )
            module['fc6']=fc6

            #fc7
            fc7=nn.Sequential(
                CBR(1024,1024,1,1,rate=1),
                nn.Dropout2d()
            )
            module['fc7']=fc7

            #score
            score_fr=nn.Conv2d(1024,num_classes,1)
            module['score_fr']=score_fr

            return module

        self.backbone=vgg16()
        self.classifier=classifier()
        self.upsampling=upsampling



    def forward(self, x):
        #backbone
        for key in self.backbone.keys():
            x=self.backbone[key](x)

        _,_,feature_map_h,feature_map_w=x.size() # BxCxHxW

        #classifier
        for key in self.classifier.keys():
            x=self.classifier[key](x)

        #Upsampling
        return F.interpolate(
            x,
            size=(feature_map_h*self.upsampling,feature_map_w*self.upsampling),
            mode='bilinear',
            align_corners=True)

#############################################################################################
class DilatedNet(nn.Module):
    """
    Some Information about DilatedNet

    conv1(rate=1) : 3x3 conv , 3x3 conv , 2x2 MaxPooling(stride=2,padding=0)
    conv2(rate=1) : 3x3 conv , 3x3 conv , 2x2 MaxPooling(stride=2,padding=0)
    conv3(rate=1) : 3x3 conv , 3x3 conv , 3x3 conv , 2x2 MaxPooling(stride=2,padding=0)
    conv4(rate=1) : 3x3 conv , 3x3 conv , 3x3 conv 
    conv5(rate=2) : 3x3 conv , 3x3 conv , 3x3 conv 
    FC6(rate=4) : 7x7 conv 
    FC7 : 1x1 conv
    Score : 1x1 conv

    <Additionally suggested>
    Basic Context Module : 3x3 conv(rate=1) ,3x3 conv(rate=1),3x3 conv(rate=2),3x3 conv(rate=4),3x3 conv(rate=8),3x3 conv(rate=16),3x3 conv(rate=1),1x1 conv(rate=1,no truncation)
    </Additionally suggested>

    Up Sampling : Deconv

    <주목>
    DeepLabV1과 구조적으론 거의 유사하나, pooling 차이에 대해 살펴보는것이 도움 될 것 같다.
    """
    def __init__(self,num_classes=12):
        super(DilatedNet, self).__init__()

        def CBR(in_channels,out_channels,kernel_size,stride,rate):
                return nn.Sequential(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=rate, # padding = rate 
                dilation=rate
                ),
                nn.ReLU(inplace=True)

            )
        def vgg16():
            module=nn.ModuleDict()
            #conv1(rate=1)
            conv1=nn.Sequential(
                CBR(in_channels=3,out_channels=64,kernel_size=3,stride=1,rate=1),
                CBR(64,64,3,1,rate=1),
                nn.MaxPool2d(kernel_size=2,stride=2,padding=0) # Original -> 1/2
                
            )
            module['conv1']=conv1

            #conv2(rate=1)
            conv2=nn.Sequential(
                CBR(64,128,3,1,rate=1),
                CBR(128,128,3,1,rate=1),
                nn.MaxPool2d(2,2,0) # Original -> 1/4
            )
            module['conv2']=conv2

            #conv3(rate=1)
            conv3=nn.Sequential(
                CBR(128,256,3,1,rate=1),
                CBR(256,256,3,1,rate=1),
                CBR(256,256,3,1,rate=1),
                nn.MaxPool2d(2,2,0) # Original -> 1/8
            )
            module['conv3']=conv3

            #conv4(rate=1)
            conv4=nn.Sequential(
                CBR(256,512,3,1,rate=1),
                CBR(512,512,3,1,rate=1),
                CBR(512,512,3,1,rate=1),
                # Not pooling : Fixed image size. Orginal -> 1/8(Fixed)
            )
            module['conv4']=conv4

            #conv5(rate=2)
            conv5=nn.Sequential(
                CBR(512,512,3,1,rate=2),
                CBR(512,512,3,1,rate=2),
                CBR(512,512,3,1,rate=2),
                # Not pooling : Fixed image size. Orginal -> 1/8(Fixed)
            )
            module['conv5']=conv5

            return module
        
        def classifier():
            module=nn.ModuleDict()
            #fc6(rate=4)
            fc6=nn.Sequential(
                nn.Conv2d(512,4096,kernel_size=7,dilation=4,padding=12),
                nn.ReLU(inplace=True),
                nn.Dropout2d()
            )
            module['fc6']=fc6

            #fc7
            fc7=nn.Sequential(
                nn.Conv2d(4096,4096,1),
                nn.ReLU(inplace=True),
                nn.Dropout2d()
            )
            module['fc7']=fc7
            
            #score
            score_fr=nn.Conv2d(4096,num_classes,1)
            module['score_fr']=score_fr 


            return module

        def basic_context_module():
            return nn.Sequential(
                CBR(num_classes,num_classes,3,1,rate=1),
                CBR(num_classes,num_classes,3,1,rate=1),
                CBR(num_classes,num_classes,3,1,rate=2),
                CBR(num_classes,num_classes,3,1,rate=4),
                CBR(num_classes,num_classes,3,1,rate=8),
                CBR(num_classes,num_classes,3,1,rate=16),
                CBR(num_classes,num_classes,3,1,rate=1),
                #No Truncation
                nn.Conv2d(num_classes,num_classes,1,1), 
            )
        
        def deconv():
            return nn.ConvTranspose2d(
                    num_classes,
                    num_classes,
                    kernel_size=16,
                    stride=8,
                    padding=4

                )# 1/8 -> x8 upsampling -> Original
            
        self.backbone=vgg16()
        self.classifier=classifier()
        self.basic_context_module=basic_context_module()
        self.deconv=deconv() 
        


    def forward(self, x):
        #backbone
        for key in self.backbone.keys():
            x=self.backbone[key](x)
        #classifier
        for key in self.classifier.keys():
            x=self.classifier[key](x)
        #basic_context_module
        x=self.basic_context_module(x)

        #deconv
        x=self.deconv(x)

        return x



#############################################################################################
class DeepLabV2(nn.Module):
    """
    Some Information about DeepLabV2

    -VGG version- 
    conv1(rate=1) : 3x3 conv , 3x3 conv , 3x3 MaxPooling
    conv2(rate=1) : 3x3 conv , 3x3 conv , 3x3 MaxPooling
    conv3(rate=1) : 3x3 conv , 3x3 conv , 3x3 conv, 3x3 MaxPooling
    conv4(rate=1) : 3x3 conv , 3x3 conv , 3x3 conv, 3x3 MaxPooling
    conv5(rate=2) : 3x3 conv , 3x3 conv , 3x3 conv, 3x3 MaxPooling
    fc6(rate=[6,12,18,24]) : 3x3 conv
    fc7 : 1x1 conv
    score : 1x1 conv 
    Up Sampling : Bi-linear Interpolation

    <주목>
    DeepLab V1  과 차이는 fc6,score,upsampling 방식이며 , ASPP(Atrous Spatial Pyramid Pooling) 적용 

    -ResNet version-


    """
    def __init__(self,num_classes=12,upsampling=8):
        super(DeepLabV2, self).__init__()
        self.upsampling=upsampling

        def CBR(in_channels,out_channels,kernel_size,stride,rate):
            return nn.Sequential(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=rate, # padding = rate 
                dilation=rate
                ),
                nn.ReLU(inplace=True)

            )
        def vgg16():
            module=nn.ModuleDict()
            #conv1(rate=1)
            conv1=nn.Sequential(
                CBR(in_channels=3,out_channels=64,kernel_size=3,stride=1,rate=1),
                CBR(64,64,3,1,rate=1),
                nn.MaxPool2d(kernel_size=3,stride=2,padding=1) # Original -> 1/2
                
            )
            module['conv1']=conv1

            #conv2(rate=1)
            conv2=nn.Sequential(
                CBR(64,128,3,1,rate=1),
                CBR(128,128,3,1,rate=1),
                nn.MaxPool2d(3,2,1) # Original -> 1/4
            )
            module['conv2']=conv2

            #conv3(rate=1)
            conv3=nn.Sequential(
                CBR(128,256,3,1,rate=1),
                CBR(256,256,3,1,rate=1),
                CBR(256,256,3,1,rate=1),
                nn.MaxPool2d(3,2,1) # Original -> 1/8
            )
            module['conv3']=conv3

            #conv4(rate=1)
            conv4=nn.Sequential(
                CBR(256,512,3,1,rate=1),
                CBR(512,512,3,1,rate=1),
                CBR(512,512,3,1,rate=1),
                nn.MaxPool2d(3,1,1) # stride = 1 -> Fixed image size.  Original -> 1/8(Fixed)
            )
            module['conv4']=conv4

            #conv5(rate=2)
            conv5=nn.Sequential(
                CBR(512,512,3,1,rate=2),
                CBR(512,512,3,1,rate=2),
                CBR(512,512,3,1,rate=2),
                nn.MaxPool2d(3,1,1), # stride = 1 -> Fixed image size. Orginal -> 1/8(Fixed)
            )
            module['conv5']=conv5


            return module
        
        def aspp(rate):
            return nn.Sequential(
                CBR(512,1024,3,1,rate=rate), #3x3 conv 
                nn.Dropout2d(),

                nn.Conv2d(1024,1024,kernel_size=1), #1x1 conv
                nn.ReLU(inplace=True),
                nn.Dropout2d(),

                nn.Conv2d(1024,num_classes,kernel_size=1) # 1x1 out
            )
        
        self.backbone=vgg16()
        #ASPP
        self.aspp_r6=aspp(rate=6)# rate 6
        self.aspp_r12=aspp(rate=12) # rate 12
        self.aspp_r18=aspp(rate=18)# rate 18
        self.aspp_r24=aspp(rate=24) # rate 24





    def forward(self, x):
        #backbone
        for key in self.backbone.keys():
            x=self.backbone[key](x)
        
        _,_,feature_map_h,feature_map_w=x.size()
        #ASPP
        out_r6=self.aspp_r6(x)
        out_r12=self.aspp_r12(x)
        out_r18=self.aspp_r18(x)
        out_r24=self.aspp_r24(x)
        x=sum([out_r6,out_r12,out_r18,out_r24])

        #Upsampling
        x=F.interpolate(
            x,
            size=(feature_map_h*self.upsampling,feature_map_w*self.upsampling),
            mode='bilinear',
            align_corners=True
        )

        return x


#############################################################################################
class DeepLabV3(nn.Module):
    """Some Information about DeepLabV3"""
    def __init__(self):
        super(DeepLabV3, self).__init__()

    def forward(self, x):

        return x


if __name__=="__main__":
    unittest.main()
