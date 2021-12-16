import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from helper_blocks import RRCNN_block, up_conv
import torchvision
# from model import count_parameters
import pdb

class Linknet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lnet = smp.Linknet(in_channels=1, encoder_name='resnet34', classes=2, activation=None)

    def forward(self, input):
        return self.lnet(input)

class DeepLabV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.deeplabv3 = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=2)
        self.deeplabv3.backbone.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)

    def forward(self, input):
        return self.deeplabv3(input)

class PSPNet(nn.Module):
    def __init__(self):
        super(PSPNet, self).__init__()
        self.psp = smp.PSPNet(in_channels=1, classes=2, activation=None)

    def forward(self, input):
        return self.psp(input)

class FCN_ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.fcn = torchvision.models.segmentation.fcn_resnet50(pretrained=False, progress=True, num_classes=2, aux_loss=None)
        self.fcn.backbone.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)

    def forward(self, input):
        return self.fcn(input)

class R2U_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=2, t=2):
        super(R2U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        # self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        # self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)

        # self.Up5 = up_conv(ch_in=1024, ch_out=512)
        # self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

        # self.Up4 = up_conv(ch_in=512, ch_out=256)
        # self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        # x4 = self.Maxpool(x3)
        # x4 = self.RRCNN4(x4)

        # x5 = self.Maxpool(x4)
        # x5 = self.RRCNN5(x5)

        # decoding + concat path
        # d5 = self.Up5(x5)
        # d5 = torch.cat((x4, d5), dim=1)
        # d5 = self.Up_RRCNN5(d5)

        # d4 = self.Up4(x4)
        # d4 = torch.cat((x3, d4), dim=1)
        # d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(x3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == '__main__':

    r2u_net= R2U_Net().to(device)
    # lnet = Linknet().to(device)           # OKAY
    # deeplabv3 = DeepLabV3().to(device)    # 39M - OKAY
    # pspnet = PSPNet().to(device)          # 21M - OKAY
    # fcn = FCN_ResNet50().to(device)       # 32M

    # params:
    # print('DeepLabv3: ', count_parameters(deeplabv3))
    # print('PSP net: ', count_parameters(pspnet))
    # print('FCN: ', count_parameters(fcn))

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    x = torch.randn(16, 1, 256, 256).to(device)
    print('params: ', count_parameters(r2u_net))
    pdb.set_trace()
    print(r2u_net(x).shape)
    exit(0)


    for m in [lnet, deeplabv3, pspnet, fcn]:
        out = m(x)
        if not isinstance(out, torch.Tensor):
            out = out['out']
        print(m.__class__.__name__, out.shape, out.sum(1)[0, 0, 0])

    pdb.set_trace()

    # print('Lnet: ', lnet(x).shape, count_parameters(lnet))
    # print('R2U net: ', r2u_net(x).shape, count_parameters(r2u_net))
    # print('deep lab: ', deeplabv3(x).shape)
    # print('psp: ', pspnet(x).shape)
    # print('fcn: ', fcn(x).shape)