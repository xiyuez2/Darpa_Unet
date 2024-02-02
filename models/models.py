import pytorch_lightning as pl
# import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from adamp import AdamP
from torchmetrics.functional import jaccard_index
import torchvision.models as models
from .res_unet import Resnet_Unet
from .vision_transformer import SwinUnet
from torch.nn.init import kaiming_normal_, constant_
import matplotlib.pyplot as plt
import numpy as np
from .superpixel_utils import *

class SegmentationModel(pl.LightningModule):
    def __init__(self, args=None):
        super().__init__()
        model_name = args.model
        n_channels = 6
        self.superpixel_weights = args.superpixel #'/u/xiyuez2/xiyuez2/Darpa_Unet/models/SpixelNet_bsd_ckpt.tar'
        self.superpixel = (len(self.superpixel_weights) > 0)
            
        if args.edge:
            n_channels += 1
        if self.superpixel:
            n_channels += 9
            self.superpixel_model = SpixelNet(batchNorm=True)
            # load ckpt to model
            self.superpixel_model.load_state_dict(torch.load(self.superpixel_weights)["state_dict"])
        else:
            self.superpixel_model = None

        if model_name == "Unet":
            self.model = UNet(n_channels=n_channels,n_classes=2) # was 6*5 when adding the sin preprocess
        elif model_name == "Unet2B":
            self.model = UNet2Branch(n_channels=n_channels,n_classes=2)
        elif model_name == "Resnet":
            self.model = Resnet_Unet(n_channels=n_channels,n_classes=2)
        elif model_name == "swin":
            self.model = SwinUnet(n_channels=n_channels,n_classes=2)
        else:
            raise NotImplementedError
        self.args = args
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'adamp':
            optimizer = AdamP(self.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-2)

        if self.args.scheduler == "reducelr":
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, mode="max", verbose=True)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/jaccard_index_value"}

        elif self.args.scheduler == "cosineanneal":
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5,
                                                                 last_epoch=-1, verbose=True)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, train_batch, batch_idx):
        if self.superpixel:
            map_img, legend_img, mask, superpixel = train_batch["map_img"],train_batch["legend_img"],train_batch["GT"],train_batch["superpixel"]
            
            superpixel_img = super_pixel_inference(downsize = 32, model = self.superpixel_model, img_= map_img)
            # for i in range(len(map_img)):
            #     superpixel_img.append(super_pixel_inference(downsize = 16, model = self.superpixel_model, img_= map_img[i]))    
            # visiualize superpixel, plot it and save image on disk
            # import matplotlib.pyplot as plt
            # import numpy as np
            # print("saving image")
            # viz_superpixel = superpixel_img #.cpu().numpy()
            # viz_superpixel = viz_superpixel[0,:,:,:]
            # # normalize superpixel to 0-1
            # viz_superpixel = (viz_superpixel - viz_superpixel.min()) / (viz_superpixel.max() - viz_superpixel.min())
            # viz_superpixel = np.transpose(viz_superpixel,(1,2,0))
            # plt.imshow(viz_superpixel)
            # plt.savefig("superpixel.png")
            # plt.close()
            
            # superpixel_img = torch.from_numpy(np.array(superpixel_img)).float().cuda()
            map_img = torch.cat((map_img, superpixel_img),axis=1)

        else:
            map_img, legend_img, mask = train_batch["map_img"],train_batch["legend_img"],train_batch["GT"]
        

        mask = mask.long()
        outputs = self.model(map_img,legend_img)
        loss = self.criterion(outputs, mask)
        jaccard_index_value = jaccard_index(outputs.argmax(dim=1), mask, task="multiclass", num_classes=2)

        self.log('train/loss', loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        self.log('train/jaccard_index_value', jaccard_index_value, on_epoch=True, on_step=True, prog_bar=True,
                 sync_dist=True)

        return {"loss": loss, "jaccard_index_value": jaccard_index_value}

    def validation_step(self, val_batch, batch_idx):
        if self.superpixel:
            map_img, legend_img, mask, superpixel = val_batch["map_img"],val_batch["legend_img"],val_batch["GT"],val_batch["superpixel"]
            
            superpixel_img = super_pixel_inference(downsize = 32, model = self.superpixel_model, img_= map_img)
            # for i in range(len(map_img)):
            #     superpixel_img.append(super_pixel_inference(downsize = 16, model = self.superpixel_model, img_= map_img[i]))    
            # superpixel_img = torch.from_numpy(np.array(superpixel_img)).float().cuda()
            map_img = torch.cat((map_img, superpixel_img), axis=1)
        else:
            map_img, legend_img, mask = val_batch["map_img"], val_batch["legend_img"], val_batch["GT"]
        mask = mask.long()

        outputs = self.model(map_img,legend_img)
        loss = self.criterion(outputs, mask)
        jaccard_index_value = jaccard_index(outputs.argmax(dim=1), mask,task="multiclass", num_classes=2)

        self.log('val/loss', loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        self.log('val/jaccard_index_value', jaccard_index_value, on_epoch=True, on_step=True, prog_bar=True,
                 sync_dist=True)

        return {"loss": loss, "jaccard_index_value": jaccard_index_value}

    def test_step(self, val_batch, batch_idx):
        if self.superpixel:
            map_img, legend_img, mask, superpixel = val_batch["map_img"].cuda(), val_batch["legend_img"].cuda(), val_batch["GT"].cuda(), val_batch["superpixel"].cuda()
            
            superpixel_img = super_pixel_inference(downsize = 32, model = self.superpixel_model, img_= map_img)
            # for i in range(len(map_img)):
            #     superpixel_img.append(super_pixel_inference(downsize = 16, model = self.superpixel_model, img_= map_img[i]))    
            # superpixel_img = torch.from_numpy(np.array(superpixel_img)).float().cuda()
            map_img = torch.cat((map_img, superpixel_img),axis=1)
        else:
            map_img, legend_img, mask = val_batch["map_img"].cuda(), val_batch["legend_img"].cuda(), val_batch["GT"].cuda()
        mask = mask.long()

        outputs = self.model(map_img,legend_img)
        # loss = self.criterion(outputs, mask)
        # jaccard_index_value = jaccard_index(outputs.argmax(dim=1), mask, num_classes=2)
        return outputs[:,-1,]




def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

def predict_param(in_planes, channel=3):
    return  nn.Conv2d(in_planes, channel, kernel_size=3, stride=1, padding=1, bias=True)

def predict_mask(in_planes, channel=9):
    return  nn.Conv2d(in_planes, channel, kernel_size=3, stride=1, padding=1, bias=True)

def predict_feat(in_planes, channel=20, stride=1):
    return  nn.Conv2d(in_planes, channel, kernel_size=3, stride=stride, padding=1, bias=True)

def predict_prob(in_planes, channel=9):
    return  nn.Sequential(
        nn.Conv2d(in_planes, channel, kernel_size=3, stride=1, padding=1, bias=True),
        nn.Softmax(1)
    )
#***********************************************************************

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1)
        )


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1)
    )

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Fuse_Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, fuse_channel = 0):
        super().__init__()
        if fuse_channel == 0:
            fuse_channel = in_channels
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2 + fuse_channel, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, map, legend):
        
        # print("debug in train",map.size(),legend.size())    
        # print("debug in train",map.min(),legend.min())
        # print("debug in train",map.max(),legend.max())
        # print("debug in train",map.mean(),legend.mean())
        # # normalize map and legend to 0-1
        # viz_map = (map - map.min()) / (map.max() - map.min())
        # viz_legend =  (map - legend.min()) / (legend.max() - legend.min())

        # save map and legend into images on disk
        # viz_map = viz_map.cpu().numpy()
        # viz_legend = viz_legend.cpu().numpy()
        # viz_map = viz_map[0,:,:,:]
        # viz_legend = viz_legend[0,:,:,:]
        # viz_map = np.transpose(viz_map,(1,2,0))
        # viz_legend = np.transpose(viz_legend,(1,2,0))
        # plt.imshow(viz_map)
        # plt.savefig("map.png")
        # plt.close()
        # plt.imshow(viz_legend)
        # plt.savefig("legend.png")
        # plt.close()

        x = torch.cat((map, legend),axis=1)
        # print(x.size(),map.size(),legend.size())
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # logits = torch.argmax(logits, dim=1)
        # print(logits.size())
        # viz_logits =  torch.argmax(logits, dim=1).cpu().numpy()
        # print(viz_logits.shape)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

class UNet2Branch(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet2Branch, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # encoder for map
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        
        # encoder for legend
        self.incl = (DoubleConv(n_channels, 64))
        self.downl1 = (Down(64, 128))
        self.downl2 = (Down(128, 256))
        self.downl3 = (Down(256, 512))
        self.downl4 = (Down(512, 1024 // factor))

        # decoder
        # self.up1 = (Fuse_Up(1024*2, 512 // factor, bilinear,fuse_channel = 512))
        
        self.fuseconv = DoubleConv(1024 * 2, 1024)
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, map, legend):
        # x = torch.cat((map, legend),axis=1)
        # print(x.size(),map.size(),legend.size())

        map1 = self.inc(map)
        map2 = self.down1(map1)
        map3 = self.down2(map2)
        map4 = self.down3(map3)
        map5 = self.down4(map4)

        legend1 = self.incl(legend)
        legend2 = self.downl1(legend1)
        legend3 = self.downl2(legend2)
        legend4 = self.downl3(legend3)
        legend5 = self.downl4(legend4)

        fused5 = torch.cat((map5,legend5),axis=1)
        # fused4 = torch.cat((map4,legend4),axis=1)
        # fused3 = torch.cat((map3,legend3),axis=1)
        # fused2 = torch.cat((map2,legend2),axis=1)
        # fused1 = torch.cat((map1,legend1),axis=1)
        fused5 = self.fuseconv(fused5)

        x = self.up1(fused5, map4)
        x = self.up2(x, map3)
        x = self.up3(x, map2)
        x = self.up4(x, map1)
        logits = self.outc(x)
        # logits = torch.argmax(logits, dim=1)
        # print(logits.size())
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        
        self.incl = torch.utils.checkpoint(self.incl)
        self.downl1 = torch.utils.checkpoint(self.downl1)
        self.downl2 = torch.utils.checkpoint(self.downl2)
        self.downl3 = torch.utils.checkpoint(self.downl3)
        self.downl4 = torch.utils.checkpoint(self.downl4)

        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class UNet_resnet18(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        
        self.base_model = models.resnet18()
        self.base_model.load_state_dict(torch.load("../input/resnet18/resnet18.pth"))
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, map, legend):
        input = torch.cat((map, legend),axis=1)
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out

class SpixelNet(nn.Module):
    expansion = 1
    def __init__(self, batchNorm=True):
        super(SpixelNet,self).__init__()

        self.batchNorm = batchNorm
        self.assign_ch = 9

        self.conv0a = conv(self.batchNorm, 3, 16, kernel_size=3)
        self.conv0b = conv(self.batchNorm, 16, 16, kernel_size=3)

        self.conv1a = conv(self.batchNorm, 16, 32, kernel_size=3, stride=2)
        self.conv1b = conv(self.batchNorm, 32, 32, kernel_size=3)

        self.conv2a = conv(self.batchNorm, 32, 64, kernel_size=3, stride=2)
        self.conv2b = conv(self.batchNorm, 64, 64, kernel_size=3)

        self.conv3a = conv(self.batchNorm, 64, 128, kernel_size=3, stride=2)
        self.conv3b = conv(self.batchNorm, 128, 128, kernel_size=3)

        self.conv4a = conv(self.batchNorm, 128, 256, kernel_size=3, stride=2)
        self.conv4b = conv(self.batchNorm, 256, 256, kernel_size=3)

        self.deconv3 = deconv(256, 128)
        self.conv3_1 = conv(self.batchNorm, 256, 128)
        self.pred_mask3 = predict_mask(128, self.assign_ch)

        self.deconv2 = deconv(128, 64)
        self.conv2_1 = conv(self.batchNorm, 128, 64)
        self.pred_mask2 = predict_mask(64, self.assign_ch)

        self.deconv1 = deconv(64, 32)
        self.conv1_1 = conv(self.batchNorm, 64, 32)
        self.pred_mask1 = predict_mask(32, self.assign_ch)

        self.deconv0 = deconv(32, 16)
        self.conv0_1 = conv(self.batchNorm, 32 , 16)
        self.pred_mask0 = predict_mask(16,self.assign_ch)

        self.softmax = nn.Softmax(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        out1 = self.conv0b(self.conv0a(x)) #5*5
        out2 = self.conv1b(self.conv1a(out1)) #11*11
        out3 = self.conv2b(self.conv2a(out2)) #23*23
        out4 = self.conv3b(self.conv3a(out3)) #47*47
        out5 = self.conv4b(self.conv4a(out4)) #95*95

        out_deconv3 = self.deconv3(out5)
        concat3 = torch.cat((out4, out_deconv3), 1)
        out_conv3_1 = self.conv3_1(concat3)

        out_deconv2 = self.deconv2(out_conv3_1)
        concat2 = torch.cat((out3, out_deconv2), 1)
        out_conv2_1 = self.conv2_1(concat2)

        out_deconv1 = self.deconv1(out_conv2_1)
        concat1 = torch.cat((out2, out_deconv1), 1)
        out_conv1_1 = self.conv1_1(concat1)

        out_deconv0 = self.deconv0(out_conv1_1)
        concat0 = torch.cat((out1, out_deconv0), 1)
        out_conv0_1 = self.conv0_1(concat0)
        mask0 = self.pred_mask0(out_conv0_1)
        prob0 = self.softmax(mask0)

        return prob0

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

def super_pixel_inference(downsize, model, img_):
    # may get 4 channel (alpha channel) for some format
    # img is of shape (3, H, W)
    b, _, H, W = img_.shape
    H_, W_  = int(np.ceil(H/16.)*16), int(np.ceil(W/16.)*16)

    # get spixel id
    n_spixl_h = int(np.floor(H_ / downsize))
    n_spixl_w = int(np.floor(W_ / downsize))

    spix_values = np.int32(np.arange(0, n_spixl_w * n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
    spix_idx_tensor_ = shift9pos(spix_values)

    spix_idx_tensor = np.repeat(
      np.repeat(spix_idx_tensor_, downsize, axis=1), downsize, axis=2)

    spixeIds = torch.from_numpy(np.tile(spix_idx_tensor, (1, 1, 1, 1))).type(torch.float).cuda()
    n_spixel =  int(n_spixl_h * n_spixl_w)

    img = img_
    img1 = img_
    ori_img = img_
    
    # compute output
    output = model(img1)

    spixel_viz = output #np.zeros((len(output), 3, H_, W_))
    # for i in range(len(output)):
    # # assign the spixel map
    #     curr_spixl_map = update_spixl_map(spixeIds, output[i].unsqueeze(0))
    #     ori_sz_spixel_map = F.interpolate(curr_spixl_map.type(torch.float), size=( H_,W_), mode='nearest').type(torch.int)

    #     mean_values = torch.tensor([0.411, 0.432, 0.45], dtype=img1[i].cuda().unsqueeze(0).dtype).view(3, 1, 1).cuda()
    #     # import pdb; pdb.set_trace()
    #     spixel_viz[i], spixel_label_map = get_spixel_image((ori_img[i] + mean_values).clamp(0, 1), ori_sz_spixel_map.squeeze(), n_spixels= n_spixel,  b_enforce_connect=True)
        
    # plt.imshow(spixel_viz.transpose(1,2,0))
    # plt.savefig("superpixel_img.png")
    # plt.close()
    # print(np.shape(spixel_viz))
    return spixel_viz