import torch
import torch.nn as nn
import math


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1,padding=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.1,inplace=True)
    )

def conv_basic(dropout, in_planes, out_planes, kernel_size=3, stride=1,padding=1):
    if dropout:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.PReLU(),
            nn.Dropout(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.PReLU()
        )




class DeepPoseCorrectorStereoFullPose(nn.Module):
    def __init__(self):
        super(DeepPoseCorrectorStereoFullPose, self).__init__()
        # conv layers: (in_channel size, out_channels size, kernel_size, stride, padding)

                # conv layers: (in_channel size, out_channels size, kernel_size, stride, padding)
        self.cnn = nn.Sequential(
            conv_basic( True, 6, 64, kernel_size=3, stride=2, padding=1),
            conv_basic( True, 64, 64, kernel_size=3, stride=2, padding=1),
            conv_basic( True, 64, 128, kernel_size=3, stride=1, padding=1),
        )
        

        self.concat_net = nn.Sequential(
            conv_basic( True, 256, 256, kernel_size=3, stride=2,padding=1),
            conv_basic( True, 256, 512, kernel_size=3, stride=2,padding=1),
            conv_basic( True, 512, 1024, kernel_size=3, stride=2,padding=1),
            conv_basic( True, 1024, 4096, kernel_size=3, stride=2,padding=1),
            conv_basic( True, 4096, 4096, kernel_size=3, stride=2,padding=1),
            conv_basic( True, 4096, 6, kernel_size=3, stride=2,padding=1),
            nn.Conv2d(6, 6, kernel_size=(1,2), stride=2,padding=0)
        ) 

        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, 0.5 / math.sqrt(n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        #Initialize last conv weights to 0 to ensure the initial transform is Identity
        self.concat_net[-1].weight.data.zero_()

    def forward(self, img_1, img_2):

        x1 = self.cnn(img_1)
        x2 = self.cnn(img_2)

        x = torch.cat((x1, x2), 1)
        y = self.concat_net(x)
        y = y.view(-1, 6)
        return y



class DeepPoseCorrectorMonoRotation(nn.Module):
    def __init__(self):
        super(DeepPoseCorrectorMonoRotation, self).__init__()

        self.cnn = nn.Sequential(
            conv_basic( True, 3, 64, kernel_size=3, stride=2, padding=1),
            conv_basic( True, 64, 64, kernel_size=3, stride=2, padding=1),
            conv_basic( True, 64, 128, kernel_size=3, stride=1, padding=1),
        )
        
        self.concat_net = nn.Sequential(
            conv_basic( True, 256, 256, kernel_size=3, stride=2,padding=1),
            conv_basic( True, 256, 512, kernel_size=3, stride=2,padding=1),
            conv_basic( True, 512, 1024, kernel_size=3, stride=2,padding=1),
            conv_basic( True, 1024, 4096, kernel_size=3, stride=2,padding=1),
            conv_basic( True, 4096, 4096, kernel_size=3, stride=2,padding=1),
            conv_basic( True, 4096, 6, kernel_size=3, stride=2,padding=1),
        ) 

        self.final_conv = nn.Conv2d(6, 3, kernel_size=(1,2), stride=2,padding=0)

        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, 0.5 / math.sqrt(n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        #Initialize last conv weights to 0 to ensure the initial transform is Identity
        self.final_conv.weight.data.zero_()
    
    def forward(self, img_1, img_2):

        x1 = self.cnn(img_1)
        x2 = self.cnn(img_2)

        x = torch.cat((x1, x2), 1)
        y = self.final_conv(self.concat_net(x))
        y = y.view(-1, 3)
        #y = self.max_norm(y)
        return y


class DeepPoseCorrectorMonoYaw(nn.Module):
    def __init__(self):
        super(DeepPoseCorrectorMonoYaw, self).__init__()

        # conv layers: (in_channel size, out_channels size, kernel_size, stride, padding)
        self.cnn = nn.Sequential(
            conv_basic( True, 3, 64, kernel_size=3, stride=2, padding=1),
            conv_basic( True, 64, 64, kernel_size=3, stride=2, padding=1),
            conv_basic( True, 64, 128, kernel_size=3, stride=1, padding=1),
        )
        
        #self.max_norm = NormLimitLayerSE3(0.1, 0.01)

        self.concat_net = nn.Sequential(
            conv_basic( True, 256, 256, kernel_size=3, stride=2,padding=1),
            conv_basic( True, 256, 512, kernel_size=3, stride=2,padding=1),
            conv_basic( True, 512, 1024, kernel_size=3, stride=2,padding=1),
            conv_basic( True, 1024, 4096, kernel_size=3, stride=2,padding=1),
            conv_basic( True, 4096, 4096, kernel_size=3, stride=2,padding=1),
            conv_basic( True, 4096, 6, kernel_size=3, stride=2,padding=1),
            nn.Conv2d(6, 1, kernel_size=(1,2), stride=2,padding=0),
        ) 
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt( 0.5 / n ))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, img_1, img_2):

        x1 = self.cnn(img_1)
        x2 = self.cnn(img_2)

        x = torch.cat((x1, x2), 1)
        y = self.concat_net(x)
        y = y.view(-1, 1)
        #y = self.max_norm(y)
        return y