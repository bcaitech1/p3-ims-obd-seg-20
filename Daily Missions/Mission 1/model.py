import torch
from torch.nn.modules import padding
from torch.nn.modules.activation import ReLU
import torchvision.models as models
import torch.nn as nn


class FCN(nn.Module):
    """
    Some Information about FCN
    VGG16 기반 
    """
    def __init__(self,version='8s',num_classes=21):
        super(FCN, self).__init__()
        assert version in ['8s','16s','32s']
        self.version=version

        def CBR(in_channels,out_channels,kernel_size,stride,padding): 
            return nn.Sequential(
                nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
                ),
                nn.ReLU(inplace=True)
            )

        # Pooling 마다 1/2 배씩 이미지 사이즈는 줄어든다. 
        

        #conv1
        self.conv1_1=CBR(3,64,3,1,1)
        self.conv1_2=CBR(64,64,3,1,1)
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)  # Origianl -> 1/2 when True, will use ceil instead of floor to compute the output shape 
        
        
        #conv2
        self.conv2_1=CBR(64,128,3,1,1)
        self.conv2_2=CBR(128,128,3,1,1)
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)  # Origianl -> 1/4 when True, will use ceil instead of floor to compute the output shape 
    
        #conv3
        self.conv3_1=CBR(128,256,3,1,1)
        self.conv3_2=CBR(256,256,3,1,1)
        self.conv3_3=CBR(256,256,3,1,1)
        self.pool3=nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)  # Origianl -> 1/8 when True, will use ceil instead of floor to compute the output shape 

        #conv4
        self.conv4_1=CBR(256,512,3,1,1)
        self.conv4_2=CBR(512,512,3,1,1)
        self.conv4_3=CBR(512,512,3,1,1)
        self.pool4=nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)  # Origianl -> 1/16 when True, will use ceil instead of floor to compute the output shape 

        #conv5
        self.conv5_1=CBR(512,512,3,1,1)
        self.conv5_2=CBR(512,512,3,1,1)
        self.conv5_3=CBR(512,512,3,1,1)
        self.pool5=nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)  # Origianl -> 1/32 when True, will use ceil instead of floor to compute the output shape 

        #fc6
        self.fc6=nn.Conv2d(512,4096,1)
        self.relu6=nn.ReLU(inplace=True)
        self.drop6=nn.Dropout2d()

        #fc7
        self.fc7=nn.Conv2d(4096,4096,1)
        self.relu7=nn.ReLU(inplace=True)
        self.drop7=nn.Dropout2d()

        # score
        self.score_fr=nn.Conv2d(4096,num_classes,kernel_size=1,stride=1)

        # Deconv 
        '''
        kernel_size=k=2s,
        stride=s=2p,
        padding=p
        '''
        if self.version == '32s':
            self.upscore32=nn.ConvTranspose2d( 
                num_classes,
                num_classes,
                kernel_size=64,
                stride=32,
                padding=16
            ) # 1/32 -> x32 upsampling -> Original 

        elif self.version == '16s':
            #Score pool4
            self.score_pool4_fr=nn.Conv2d(
                512,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0

            )
            self.upscore2=nn.ConvTranspose2d(
                num_classes,
                num_classes,
                kernel_size=4,
                stride=2,
                padding=1,
            ) # 1/32 -> x2 upsampling -> 1/16

            self.upscore16=nn.ConvTranspose2d(
                num_classes,
                num_classes,
                kernel_size=32,
                stride=16,
                padding=8

            ) # 1/16 -> x16 upsampling -> Original

        elif self.version == '8s':
            #Score pool3  
            self.score_pool3_fr=nn.Conv2d( 
                256,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0
                
            )

            #Score pool4
            self.score_pool4_fr=nn.Conv2d(
                512,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0

            )

            self.upscore2=nn.ConvTranspose2d(
                num_classes,
                num_classes,
                kernel_size=4,
                stride=2,
                padding=1,
            ) # 1/32 -> x2 upsampling -> 1/16

            self.upscore2_pool4=nn.ConvTranspose2d(
                num_classes,
                num_classes,
                kernel_size=4,
                stride=2,
                padding=1,
            ) # 1/16 -> x2 upsampling -> 1/8

            self.upscore8=nn.ConvTranspose2d(
                num_classes,
                num_classes,
                kernel_size=16,
                stride=8,
                padding=4,
            ) # 1/8 -> x8 upsampling -> Original

        


    def forward(self, x):
        h=self.conv1_1(x)
        h=self.conv1_2(h)
        h=self.pool1(h)

        h=self.conv2_1(h)
        h=self.conv2_2(h)
        h=self.pool2(h)

        h=self.conv3_1(h)
        h=self.conv3_2(h)
        h=self.conv3_3(h)
        pool3=h=self.pool3(h)
        if self.version == '8s':
            score_pool3c=self.score_pool3_fr(pool3)
        

        h=self.conv4_1(h)
        h=self.conv4_2(h)
        h=self.conv4_3(h)
        pool4=h=self.pool4(h)
        if self.version in ['8s','16s']:
            score_pool4c=self.score_pool4_fr(pool4)

        h=self.conv5_1(h)
        h=self.conv5_2(h)
        h=self.conv5_3(h)
        h=self.pool5(h)

        h=self.fc6(h)
        h=self.drop6(h)

        h=self.fc7(h)
        h=self.drop7(h)

        h=self.score_fr(h)

        if self.version == '32s':
            upscore32=self.upscore32(h) # 1/32 -> x32 upsampling -> Original
            return upscore32

        elif self.version == '16s':
            #Up Score I
            upscore2=self.upscore2(h) # 1/32 -> x2 upsampling ->  1/16
            h=score_pool4c+upscore2 # Skip Connection
            #Up Score II
            upscore16=self.upscore16(h)  # 1/16 -> x2 upsampling -> Original
            return upscore16


        elif self.version == '8s':
            #Up Score I
            upscore2=self.upscore2(h) # 1/32 -> x2 upsampling -> 1/16
            h=score_pool4c+upscore2 # Skip Connection

            #Up Score II
            upscore2_pool4c=self.upscore2_pool4(h) # 1/16 -> x2 upsampling -> 1/8
            h=score_pool3c+upscore2_pool4c # Skip Connection 

            #Up Score III
            upscore8=self.upscore8(h)
            return upscore8



if __name__=='__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = FCN(version='16s',num_classes=12)
    x = torch.randn([1, 3, 512, 512])
    print("input shape : ", x.shape)
    out = model(x).to(device)
    print("output shape : ", out.size())

    