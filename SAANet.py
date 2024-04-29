
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary
from model.sync_batchnorm import SynchronizedBatchNorm2d
from torchvision.models import resnet34, resnet50, resnet101, resnet152, resnet18
from model.batchnorm import SynchronizedBatchNorm2d



class SAANet(nn.Module):
    # def __init__(self, backbone,
    #              pretrained=True,
    #              ResNet34M= False,
    #              classes=11):
    def __init__(self, backbone,  sync_bn=True, pretrained=True, ResNet34M= False, criterion=nn.CrossEntropyLoss(ignore_index=255), classes = 24):
        super(SSFPN, self).__init__()
        self.ResNet34M = ResNet34M
        self.backbone = backbone
        self.criterion = criterion

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        if backbone.lower() == "resnet18":
            encoder = resnet18(pretrained=pretrained)
        elif backbone.lower() == "resnet34":
            encoder = resnet34(pretrained=pretrained)
        elif backbone.lower() == "resnet50":
            encoder = resnet50(pretrained=pretrained)
        elif backbone.lower() == "resnet101":
            encoder = resnet101(pretrained=pretrained)
        elif backbone.lower() == "resnet152":
            encoder = resnet152(pretrained=pretrained)
        else:
            raise NotImplementedError("{} Backbone not implemented".format(backbone))

        self.out_channels = [32,64,128,256,512,1024,2048]
        # self.conv1 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool)
        self.conv1_x = encoder.conv1
        self.bn1 = encoder.bn1
        self.relu = encoder.relu
        self.maxpool = encoder.maxpool
        self.conv2_x = encoder.layer1  # 1/4
        self.conv3_x = encoder.layer2  # 1/8
        self.conv4_x = encoder.layer3  # 1/16
        self.conv5_x = encoder.layer4  # 1/32

        # if backbone in ['resnet50','resnet101','resnet152']:
        self.down2 = conv_block(self.out_channels[-4], self.out_channels[1], 3, 1, 1, 1, 1, bn_act=True)
        self.down3 = conv_block(self.out_channels[-3], self.out_channels[2], 3, 1, 1, 1, 1, bn_act=True)
        self.down4 = conv_block(self.out_channels[-2], self.out_channels[3], 3, 1, 1, 1, 1, bn_act=True)
        self.down5 = conv_block(self.out_channels[-1], self.out_channels[4], 3, 1, 1, 1, 1, bn_act=True)

        self.fab = nn.Sequential(
            conv_block(self.out_channels[4],
                       self.out_channels[4]//2,
                       kernel_size = 3,
                       stride= 1,
                       padding=1,
                       group=self.out_channels[4]//2,
                       dilation=1,
                       bn_act=True),
                       nn.Dropout(p=0.3))
        self.ciam =CIAM(self.out_channels[4]//2)  #

        self.cfgb = nn.Sequential(
            conv_block(self.out_channels[4],
                       self.out_channels[4],
                       kernel_size =3,
                       stride= 2,
                       padding = 1,
                       group=self.out_channels[4],
                       dilation=1,
                       bn_act=True),
                       nn.Dropout(p=0.3))

        self.decoder = DecoderBlock(self.out_channels[4], self.out_channels[4], BatchNorm) #
        self.decoderdowm = conv_block(self.out_channels[4], self.out_channels[4],3,2,padding=1) #


        self.gfu4 = GlobalFeatureUpsample(self.out_channels[3], self.out_channels[3], self.out_channels[3])
        self.gfu3 = GlobalFeatureUpsample(self.out_channels[2], self.out_channels[3], self.out_channels[2])
        self.gfu2 = GlobalFeatureUpsample(self.out_channels[1], self.out_channels[2], self.out_channels[1])
        self.gfu1 = GlobalFeatureUpsample(self.out_channels[0], self.out_channels[1], self.out_channels[0])


        self.apf1 = PyrmidFusionNet(self.out_channels[4], self.out_channels[4], self.out_channels[3], classes=classes)
        self.apf2 = PyrmidFusionNet(self.out_channels[3], self.out_channels[3], self.out_channels[2], classes=classes)
        self.apf3 = PyrmidFusionNet(self.out_channels[2], self.out_channels[2], self.out_channels[1], classes=classes)
        self.apf4 = PyrmidFusionNet(self.out_channels[1], self.out_channels[1], self.out_channels[0], classes=classes)



        self.classifier = SegHead(self.out_channels[0], classes)

    def forward(self, x, y=None):
        B, C, H, W = x.size()
        x = self.conv1_x(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x = self.maxpool(x1)
        if self.ResNet34M:
            x2 = self.conv2_x(x1)
        else:
            x2 = self.conv2_x(x)
        x3 = self.conv3_x(x2)
        x4 = self.conv4_x(x3)
        x5 = self.conv5_x(x4)

        if self.backbone in ['resnet50', 'resnet101', 'resnet152']:
            x2 = self.down2(x2)
            x3 = self.down3(x3)
            x4 = self.down4(x4)
            x5 = self.down5(x5)

        # print(x5.size())  #torch.Size([6, 512, 16, 16])
        # CFGB = self.cfgb(x5) #torch.Size([6, 512, 8, 8])
        # print(CFGB.size())
        cfgb1 = self.decoder(x5) #torch.Size([6, 512, 16, 16])
        # print(cfgb1.size())
        CFGB =self.decoderdowm(cfgb1) #torch.Size([6, 512, 8, 8])
        # print(cfgb1d.size())

        APF1, cls1 = self.apf1(CFGB, x5)
        APF2, cls2 = self.apf2(APF1, x4)
        APF3, cls3 = self.apf3(APF2, x3)
        APF4, cls4 = self.apf4(APF3, x2)

        # print("WWWWWW")
        FAB = self.fab(x5) #torch.Size([6, 256, 16, 16])
        # print(FAB.size()) #torch.Size([6, 256, 16, 16])
        # fab = self.ciam(x5)
        # print(fab.size())



        dec5 = self.gfu4(APF1, FAB)  
        dec4 = self.gfu3(APF2, dec5)
        dec3 = self.gfu2(APF3, dec4)
        dec2 = self.gfu1(APF4, dec3)

        classifier = self.classifier(dec2)

        sup1 = F.interpolate(cls1, size=(H, W), mode="bilinear", align_corners=True)
        sup2 = F.interpolate(cls2, size=(H, W), mode="bilinear", align_corners=True)
        sup3 = F.interpolate(cls3, size=(H, W), mode="bilinear", align_corners=True)
        sup4 = F.interpolate(cls4, size=(H, W), mode="bilinear", align_corners=True)
        predict = F.interpolate(classifier, size=(H, W), mode="bilinear", align_corners=True)

        if self.training:
            main_loss = self.criterion(predict, y)
            return predict.max(1)[1], main_loss, main_loss
        else:
            return predict

        return predict
        # if self.training:
        #     return predict, sup1, sup2, sup3, sup4
        # else:
        #     return predict

class PyrmidFusionNet(nn.Module):
    def __init__(self, channels_high, channels_low, channel_out, classes=11):
        super(PyrmidFusionNet, self).__init__()

        self.lateral_low = conv_block(channels_low, channels_high, 1, 1, bn_act=True, padding=0)

        self.conv_low = conv_block(channels_high, channel_out, 3, 1, bn_act=True, padding=1)
        self.sa = SpatialAttention(channel_out, channel_out)

        self.conv_high = conv_block(channels_high, channel_out, 3, 1, bn_act=True, padding=1)
        self.ca = ChannelWise(channel_out)

        self.FRB = nn.Sequential(
            conv_block(2 * channels_high, channel_out, 1, 1, bn_act=True, padding=0),
            conv_block(channel_out, channel_out, 3, 1, bn_act=True, group=1, padding=1))

        self.classifier = nn.Sequential(
            conv_block(channel_out, channel_out, 3, 1, padding=1, group=1, bn_act=True),
            nn.Dropout(p=0.15),
            conv_block(channel_out, classes, 1, 1, padding=0, bn_act=False))
        self.apf = conv_block(channel_out, channel_out, 3, 1, padding=1, group=1, bn_act=True)

    def forward(self, x_high, x_low):
        _, _, h, w = x_low.size()

        lat_low = self.lateral_low(x_low)

        high_up1 = F.interpolate(x_high, size=lat_low.size()[2:], mode='bilinear', align_corners=False)

        concate = torch.cat([lat_low, high_up1], 1)
        concate = self.FRB(concate)

        conv_high = self.conv_high(high_up1)
        conv_low = self.conv_low(lat_low)

        sa = self.sa(concate)
        ca = self.ca(concate)

        mul1 = torch.mul(sa, conv_high)
        mul2 = torch.mul(ca, conv_low)

        att_out = mul1 + mul2

        sup = self.classifier(att_out)
        APF = self.apf(att_out)
        return APF,sup


class GlobalFeatureUpsample(nn.Module):
    def __init__(self, low_channels, in_channels, out_channels):
        super(GlobalFeatureUpsample, self).__init__()

        self.conv1 = conv_block(low_channels, out_channels, kernel_size=1, stride=1, padding=0, bn_act=True)
        self.conv2 = nn.Sequential(
            conv_block(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bn_act=False),
            nn.ReLU(inplace=True))
        self.conv3 = conv_block(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bn_act=True)

    def forward(self, x_gui, y_high):
        h, w = x_gui.size(2), x_gui.size(3)
        y_up = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(y_high)
        x_gui = self.conv1(x_gui)
        y_up = F.avg_pool2d(self.conv2(y_up), (1, 1))
        out = y_up + x_gui

        return self.conv3(out)



class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=(1, 1), group=1, bn_act=False,
                 bias=False):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=group, bias=bias)
        self.bn = SynchronizedBatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=False)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.act(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class SegHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegHead, self).__init__()

        self.fc = conv_block(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        return self.fc(x)


class SpatialAttention(nn.Module):
    def __init__(self, in_ch, out_ch, droprate=0.15):
        super(SpatialAttention, self).__init__()
        self.conv_sh = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.bn_sh1 = nn.BatchNorm2d(in_ch)
        self.bn_sh2 = nn.BatchNorm2d(in_ch)
        self.conv_res = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.drop = droprate
        self.fuse = conv_block(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bn_act=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()

        mxpool = F.max_pool2d(x, [h, 1])  # .view(b,c,-1).permute(0,2,1)
        mxpool = F.conv2d(mxpool, self.conv_sh.weight, padding=0, dilation=1)
        mxpool = self.bn_sh1(mxpool)

        avgpool = F.avg_pool2d(x, [h, 1])  # .view(b,c,-1)
        avgpool = F.conv2d(avgpool, self.conv_sh.weight, padding=0, dilation=1)
        avgpool = self.bn_sh2(avgpool)

        att = torch.softmax(torch.mul(mxpool, avgpool), 1)
        attt1 = att[:, 0, :, :].unsqueeze(1)
        attt2 = att[:, 1, :, :].unsqueeze(1)

        fusion = attt1 * avgpool + attt2 * mxpool
        out = F.dropout(self.fuse(fusion), p=self.drop, training=self.training)

        # out = out.expand(residual.shape[0],residual.shape[1],residual.shape[2],residual.shape[3])
        out = F.relu(self.gamma * out + (1 - self.gamma) * x)
        return out


class ChannelWise(nn.Module):
    def __init__(self, channel, reduction=4):
        super(ChannelWise, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_pool = nn.Sequential(
            conv_block(channel, channel // reduction, 1, 1, padding=0, bias=False), nn.ReLU(inplace=False),
            conv_block(channel // reduction, channel, 1, 1, padding=0, bias=False), nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_pool(y)

        return x * y
# 修改

# 全局
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, BatchNorm, inp=False):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.bn1 = BatchNorm(in_channels // 4)
        self.relu1 = nn.ReLU()
        self.inp = inp

        self.deconv1 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        )
        self.deconv2 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv3 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv4 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        )

        self.bn2 = BatchNorm(in_channels // 4 + in_channels // 4)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels // 4 + in_channels // 4, n_filters, 1)
        self.bn3 = BatchNorm(n_filters)
        self.relu3 = nn.ReLU()

        self._init_weight()

    def forward(self, x, inp = False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)
        if self.inp:
            x = F.interpolate(x, scale_factor=2)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)
#

#局部精细
class eca_layer(nn.Module):
    def __init__(self, channel, k_size):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.k_size = k_size
        self.conv = nn.Conv1d(channel, channel, kernel_size=k_size, bias=False, groups=channel)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = nn.functional.unfold(y.transpose(-1, -3), kernel_size=(1, self.k_size), padding=(0, (self.k_size - 1) // 2))
        y = self.conv(y.transpose(-1, -2)).unsqueeze(-1)
        y = self.sigmoid(y)
        x = x * y.expand_as(x)
        return x

class SENet_Block(nn.Module):
    def __init__(self, ch_in, reduction=8):
        super(SENet_Block, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class MaskPredictor(nn.Module):
    def __init__(self, in_channels, wn=lambda x: torch.nn.utils.weight_norm(x)):
        super(MaskPredictor, self).__init__()
        self.spatial_mask = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=1, bias=False)

    def forward(self, x):
        spa_mask = self.spatial_mask(x)
        spa_mask = F.gumbel_softmax(spa_mask, tau=1, hard=True, dim=1)
        return spa_mask


class RIFU(nn.Module):
    def __init__(self, n_feats, reduction=8, wn=lambda x: torch.nn.utils.weight_norm(x)):
        super(RIFU, self).__init__()
        self.CA = eca_layer(n_feats, k_size=3)
        self.SE = SENet_Block(n_feats, reduction=reduction)  #

        self.MaskPredictor = MaskPredictor(n_feats * 8 // 8)

        self.k = nn.Sequential(
            wn(nn.Conv2d(n_feats * 8 // 8, n_feats * 8 // 8, kernel_size=3, padding=1, stride=1, groups=1)),
            nn.LeakyReLU(0.05),
            )

        self.k1 = nn.Sequential(
            wn(nn.Conv2d(n_feats * 8 // 8, n_feats * 8 // 8, kernel_size=3, padding=1, stride=1, groups=1)),
            nn.LeakyReLU(0.05),
            )

        self.res_scale = Scale(1)
        self.x_scale = Scale(1)

    def forward(self, x):
        res = x
        x = self.k(x)

        MaskPredictor = self.MaskPredictor(x)
        mask = (MaskPredictor[:, 1, ...]).unsqueeze(1)
        x = x * (mask.expand_as(x))

        x1 = self.k1(x)
        x2 = self.CA(x1)
        print(x2.size)
        x3 = self.SE(x1)
        print(x3.size())
        print("LLLLLLL")
        out = self.x_scale(x2) + self.res_scale(res)

        return out

class CIAM(nn.Module):
    def __init__(self, n_feats, wn=lambda x: torch.nn.utils.weight_norm(x)):
        super(CIAM, self).__init__()
        pooling_r = 2
        med_feats = n_feats // 1
        self.k1 = nn.Sequential(
            nn.ConvTranspose2d(n_feats, n_feats * 4 // 3, kernel_size=pooling_r, stride=pooling_r, padding=0, groups=1,
                               bias=True),
            nn.LeakyReLU(0.05),
            nn.Conv2d(n_feats * 4 // 3, n_feats, kernel_size=1, stride=2, padding=0, groups=1),
            )

        self.sig = nn.Sigmoid()

        self.k3 = RIFU(n_feats)

        self.k4 = RIFU(n_feats)

        self.k5 = RIFU(n_feats)

        self.res_scale = Scale(1)
        self.x_scale = Scale(1)

    def forward(self, x):
        identity = x
        _, _, H, W = identity.shape
        x1_1 = self.k3(x)
        x1 = self.k4(x1_1)

        x1_s = self.sig(self.k1(x) + x)
        x1 = self.k5(x1_s * x1)

        out = self.res_scale(x1) + self.x_scale(identity)

        return out

if __name__ == "__main__":
    input1 = torch.rand(2, 3, 360, 480)
    model = SSFPN("resnet18",ResNet34M=False)
    summary(model, torch.rand((2, 3, 360, 480)))

    # python train_cityscapes.py --dataset camvid --model MSFFNet --batch_size 4 --max_epochs 300 --train_type trainval --lr 1e-3
