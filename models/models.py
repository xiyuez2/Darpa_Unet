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

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
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


class SegmentationModel(pl.LightningModule):
    def __init__(self, args=None):
        super().__init__()
        model_name = args.model
        if model_name == "Unet":
            self.model = UNet(n_channels=7,n_classes=2) # was 6*5 when adding the sin preprocess
        elif model_name == "Unet2B":
            self.model = UNet2Branch(n_channels=7,n_classes=2)
        elif model_name == "Resnet":
            self.model = Resnet_Unet(n_channels=7,n_classes=2)
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
        map_img, legend_img, mask = val_batch["map_img"],val_batch["legend_img"],val_batch["GT"]
        mask = mask.long()

        outputs = self.model(map_img,legend_img)
        loss = self.criterion(outputs, mask)
        jaccard_index_value = jaccard_index(outputs.argmax(dim=1), mask,task="multiclass", num_classes=2)

        self.log('val/loss', loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        self.log('val/jaccard_index_value', jaccard_index_value, on_epoch=True, on_step=True, prog_bar=True,
                 sync_dist=True)

        return {"loss": loss, "jaccard_index_value": jaccard_index_value}

    def test_step(self, val_batch, batch_idx):
        map_img, legend_img, mask = val_batch["map_img"].cuda(),val_batch["legend_img"].cuda(),val_batch["GT"].cuda()
        mask = mask.long()

        outputs = self.model(map_img,legend_img)
        # loss = self.criterion(outputs, mask)
        # jaccard_index_value = jaccard_index(outputs.argmax(dim=1), mask, num_classes=2)

        return outputs[:,-1,] #{"loss": loss, "jaccard_index_value": jaccard_index_value}
    


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
        input = torch.cat((map, legend),axis=0)
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

