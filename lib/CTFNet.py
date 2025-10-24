import torch
import torch.nn as nn
import torch.nn.functional as F
from .CTFNet_part import convnext_large, Decoder, Refiner
class CTFNet(nn.Module):
    def __init__(self, channels=[192,384,768,1536]):
        super(CTFNet, self).__init__()
        self.backbone = convnext_large(pretrained=True, in_22k=False)
        print("---------------Pretained Model Load Success!!-------------")
        self.decoder = Decoder(channels)
        self.refiner1 = Refiner(channels[3])
        self.refiner2 = Refiner(channels[2])
        self.refiner3 = Refiner(channels[1])
        self.refiner4 = Refiner(channels[0])
    def forward(self, x):
        x0, x1, x2, x3 = self.backbone(x)
        s0 = self.decoder([x0, x1, x2, x3])
        
        s1 = self.refiner1(x3, s0)
        s2 = self.refiner2(x2, s1)
        s3 = self.refiner3(x1, s2)
        s4 = self.refiner4(x0, s3)
        res_0 = F.interpolate(s0, scale_factor=4, mode='bilinear') 
        res_1 = F.interpolate(s1, scale_factor=32, mode='bilinear') 
        res_2 = F.interpolate(s2, scale_factor=16, mode='bilinear') 
        res_3 = F.interpolate(s3, scale_factor=8, mode='bilinear') 
        res_4 = F.interpolate(s4, scale_factor=4, mode='bilinear') 

        return [res_0, res_1, res_2, res_3, res_4]


if __name__ == '__main__':
    
    model = CTFNet().to('cuda:0')
    import torch
    input = torch.randn(1, 3, 256, 256).to('cuda:0')
    s0,s1,s2,s3,s4 = model(input)
    print(s0.shape, s1.shape,s2.shape, s3.shape, s4.shape)