import random
import torch.utils.data
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import pdb
from torchsummary import summary
import time
import re
os.environ['KMP_DUPLICATE_LIB_OK']='True'
try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None
from ct_dataset import COVID19_CT_dataset, train_val_splits, HoriFlip, Affine
import torchvision
from modules import UpsamplerAttention, DecoderAttention, UpsamplerAttentionParallel
from std_models import Linknet, DeepLabV3, PSPNet, FCN_ResNet50, R2U_Net

lr = 0.01
beta1 = 0.9
ngpu = 2

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class UpsamplerLocalAttention(nn.Module):
    def __init__(self, context_channels, feat_channels, num_contexts, target = 'context'):
        super(UpsamplerLocalAttention, self).__init__()
        self.context_channels = context_channels
        self.feat_channels = feat_channels
        self.num_contexts = num_contexts
        self.W_context = nn.Conv2d(context_channels, context_channels, 1, 1, 0, bias = True) #nn.Linear(context_channels, feat_channels)
        self.W_feat = nn.Conv2d(feat_channels, context_channels, 1, 1, 0, bias = True)
        self.values_conv = nn.Conv2d(feat_channels, context_channels, 3, 1, 1, bias = True)
        self.upsample = nn.Upsample(scale_factor=2, mode = 'nearest')

    def forward(self, contexts, mainstream):
        '''
        contexts: is a list of context tensors
        mainstream: is a mainstream tensor.
        '''
        batch_size, _, spatial_h, spatial_w = contexts[0].shape
        queries = self.W_feat(mainstream)

        contexts_keys = self.W_context(torch.stack(contexts, dim=1) \
                            .view(batch_size * (self.num_contexts), self.context_channels, spatial_h, spatial_w)) \
                            .view(batch_size, self.num_contexts, self.context_channels, spatial_h, spatial_w) \
                            .permute(0, 3, 4, 1, 2)
                    # assume each context of shape B x 16 x 16 x 4 x 512
        values = self.values_conv(mainstream)

        queries_nearest_upsample = self.upsample(queries).unsqueeze(dim=-1).permute(0, 2, 3, 1, 4)   # assume each context of shape B x 16 x 16 x 512 x 1
        values_nearest_upsample = self.upsample(values)

        attn_weights = torch.matmul(contexts_keys, queries_nearest_upsample).transpose(dim0=3, dim1=4)     # B x 16 x 16 x 4 x 1 transposed to b x 16 x 16 x 1 x 4
        softmax_attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attention_output = torch.sum(torch.mul(torch.stack(contexts, dim=-1).permute(0, 2, 3, 1, 4), softmax_attn_weights), dim=-1, keepdim=False)  # b x 16 x 16 x 512

        additive_attn_output = attention_output.permute(0, 3, 1, 2) + values_nearest_upsample

        # get the channel axis to 1.
        return additive_attn_output


class DecoderAttentionParallel(nn.Module):
    def __init__(self, enc_channels, dec_channels, num_contexts):
        super(DecoderAttentionParallel, self).__init__()
        self.enc_channels = enc_channels
        self.dec_channels = dec_channels
        self.num_contexts = num_contexts
        self.W_encoder = nn.Conv2d(enc_channels, enc_channels, 1, 1, 0, bias = True)
        self.W_decoder = nn.Conv2d(dec_channels, enc_channels, 1, 1, 0, bias = True)
        self.agg_encoder = nn.Conv2d(enc_channels, 1, 1, 1, 0, bias = True)
        self.attention_layer = nn.Conv2d((num_contexts + 1) * enc_channels, enc_channels, 1, 1, 0, bias = True)
        self.values_conv = nn.Conv2d(dec_channels, enc_channels, 3, 1, 1, bias = True)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace = True)
        self.zero_pad = torch.nn.ZeroPad2d(1)

        self.get_neighborhood = nn.Unfold(kernel_size=(3, 3), stride = 1, padding=1)

        # func = lambda x: torch.stack([ torch.stack([x[:, :, i-1:i+2, j-1:j+2] for j in range(1, spatial_w-1)], dim=-1) for i in range(1, spatial_h-1) ], dim=-1)

    def forward(self, contexts, decoded_features):
        batch_size, _, spatial_h, spatial_w = contexts[0].shape     # b x 512 x 16 x 16
        queries = self.W_decoder(decoded_features)

        contexts_stacked = torch.stack(contexts, dim=1).view(batch_size * self.num_contexts, self.enc_channels, spatial_h, spatial_w)
                                    # b*3 x 512 x 16 x 16
        contexts_keys = self.W_encoder(contexts_stacked)  # b*3 x 512 x 16 x 16
        values = self.values_conv(decoded_features)   # b x 512 x 16 x 16

        # unfold will convert context_keys to b*3 x 512*9 x 256
        contexts_base_neighborhood = self.get_neighborhood(contexts_stacked)  # what is expected is this: b*3 x 512 x 3 x 3 x 16 x 16
        contexts_keys_neighborhood = self.get_neighborhood(contexts_keys)   # expected: b*3 x 512 x 3 x 3 x 16 x 16

        view_neighbourhood = lambda x: x.view(batch_size, self.num_contexts, self.enc_channels, 9, spatial_h, spatial_w) \
                                .transpose(dim0=2, dim1=3) \
                                .reshape(batch_size, self.num_contexts * 9, self.enc_channels, spatial_h, spatial_w)    # is of shape b x 3*9 x 512 x 16 x 16

        contexts_base_neighborhood = view_neighbourhood(contexts_base_neighborhood)  # b x 3*9 x 512 x 16 x 16
        contexts_keys_neighborhood = view_neighbourhood(contexts_keys_neighborhood)

        attn = torch.tanh(contexts_keys_neighborhood + queries.unsqueeze(dim = 1)).reshape(batch_size * self.num_contexts * 9, self.enc_channels, spatial_h, spatial_w)
                                    # b*3*9 x 512 x 16 x 16
        # pdb.set_trace()
        attn_weights = self.agg_encoder(attn)  # b*3*9 x 1 x 16 x 16
        softmax_attn_weights = nn.functional.softmax(attn_weights.view(batch_size, self.num_contexts, 9, 1, spatial_h, spatial_w), dim=2)
                                    # b x 3 x 9 x 1 x 16 x 16

        attended = torch.sum(torch.mul(contexts_base_neighborhood.view(batch_size, self.num_contexts, 9, self.enc_channels, spatial_h, spatial_w),
                                       softmax_attn_weights),
                             dim=2,
                             keepdim = False)
        # b x 3 x 9 x 512 x 16 x 16 -> summed across dim =2  gives b x 3 x 512 x 16 x 16

        attended = attended.view(batch_size, self.num_contexts*self.enc_channels, spatial_h, spatial_w)
                # -> reshaped to b x 3*512 x 16 x 16
        cat_attended = torch.cat([values, attended], dim=1)
        return self.leaky_relu(self.attention_layer(cat_attended))


class BaselineFCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode0 = nn.Conv2d(1, 32, 3, 1, 1, bias=False)  # 128
        self.encode1 = nn.Conv2d(32, 64, 3, 2, 1, bias=False)        # 128
        self.encode2 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)      # 64
        self.encode3 = nn.Conv2d(128, 256, 3, 2, 1, bias=False)     # 32
        self.encode4 = nn.Conv2d(256, 512, 3, 2, 1, bias=False)     # 16
        self.encode5 = nn.Conv2d(512, 512, 3, 2, 1, bias=False)    # 8

        self.decode1 = nn.Conv2d(1024, 512, 3, 1, 1, bias=False)
        self.decode2 = nn.Conv2d(512, 256, 3, 1, 1, bias=False)
        self.decode3 = nn.Conv2d(256, 128, 3, 1, 1, bias=False)
        self.decode4 = nn.Conv2d(128, 64, 3, 1, 1, bias=False)
        self.decode5 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.classifier_conv = nn.Conv2d(64, 2, 3, 1, 1, bias=False)

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        self.batch_norm2 = nn.BatchNorm2d(2, track_running_stats=True)
        self.batch_norm32 = nn.BatchNorm2d(32, track_running_stats=True)
        self.batch_norm64 = nn.BatchNorm2d(64, track_running_stats=True)
        self.batch_norm128 = nn.BatchNorm2d(128, track_running_stats=True)
        self.batch_norm256 = nn.BatchNorm2d(256, track_running_stats=True)
        self.batch_norm512 = nn.BatchNorm2d(512, track_running_stats=True)
        self.batch_norm1024 = nn.BatchNorm2d(1024, track_running_stats=True)

        self.upsample_by2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def conv_layer(self, input, conv, batch_norm, act, upsample = False):
        if upsample:
            input = self.upsample_by2(input)
        return batch_norm(act(conv(input)))

    def forward(self, input):
        x = self.conv_layer(input, self.encode0, self.batch_norm32, self.leaky_relu)
        x = self.conv_layer(x, self.encode1, self.batch_norm64, self.leaky_relu)
        x = self.conv_layer(x, self.encode2, self.batch_norm128, self.leaky_relu)
        x = self.conv_layer(x, self.encode3, self.batch_norm256, self.leaky_relu)
        x = self.conv_layer(x, self.encode4, self.batch_norm512, self.leaky_relu)
        encoded = self.conv_layer(x, self.encode5, self.batch_norm512, self.leaky_relu)

        z = self.conv_layer(encoded, self.decode1, self.batch_norm512, self.leaky_relu, upsample=True)
        z = self.conv_layer(z, self.decode2, self.batch_norm256, self.leaky_relu, upsample=True)
        z = self.conv_layer(z, self.decode3, self.batch_norm128, self.leaky_relu, upsample=True)
        z = self.conv_layer(z, self.decode4, self.batch_norm64, self.leaky_relu, upsample=True)
        z = self.conv_layer(z, self.decode5, self.batch_norm64, self.leaky_relu, upsample=True)

        z = self.conv_layer(z, self.classifier_conv, self.batch_norm2, self.leaky_relu, upsample=False)

        return z


class AdditiveAttentionFCN(BaselineFCN):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        x0 = self.conv_layer(input, self.encode0, self.batch_norm32, self.leaky_relu)
        x1 = self.conv_layer(x0, self.encode1, self.batch_norm64, self.leaky_relu)
        x2 = self.conv_layer(x1, self.encode2, self.batch_norm128, self.leaky_relu)
        x3 = self.conv_layer(x2, self.encode3, self.batch_norm256, self.leaky_relu)
        x4 = self.conv_layer(x3, self.encode4, self.batch_norm512, self.leaky_relu)
        encoded = self.conv_layer(x4, self.encode5, self.batch_norm512, self.leaky_relu)

        z = self.conv_layer(encoded, self.decode1, self.batch_norm512, self.leaky_relu, upsample=True) + x4
        z = self.conv_layer(z, self.decode2, self.batch_norm256, self.leaky_relu, upsample=True) + x3
        z = self.conv_layer(z, self.decode3, self.batch_norm128, self.leaky_relu, upsample=True) + x2
        z = self.conv_layer(z, self.decode4, self.batch_norm64, self.leaky_relu, upsample=True) + x1
        z = self.conv_layer(z, self.decode5, self.batch_norm64, self.leaky_relu, upsample=True)

        z = self.conv_layer(z, self.classifier_conv, self.batch_norm2, self.leaky_relu, upsample=False)
        return z


class AdditiveAttentionEdgeDecoder(BaselineFCN):
    def __init__(self):
        super().__init__()
        self.edge_prelim_conv = nn.Conv2d(1, 64, 3, 1, 1, bias=False)  # 128
        self.edge1_conv = nn.Conv2d(64, 64, 3, 2, 1, bias=False)  # 128
        self.edge2_conv = nn.Conv2d(64, 128, 3, 2, 1, bias=False)  # 64

    def forward(self, input):
        input, edge_map = input
        edge_prelim = self.conv_layer(edge_map, self.edge_prelim_conv, self.batch_norm64, self.leaky_relu)
        edge1 = self.conv_layer(edge_prelim, self.edge1_conv, self.batch_norm64, self.leaky_relu)
        edge2 = self.conv_layer(edge1, self.edge2_conv, self.batch_norm128, self.leaky_relu)

        # pdb.set_trace()
        x0 = self.conv_layer(input, self.encode0, self.batch_norm32, self.leaky_relu)
        x1 = self.conv_layer(x0, self.encode1, self.batch_norm64, self.leaky_relu)
        x2 = self.conv_layer(x1, self.encode2, self.batch_norm128, self.leaky_relu)
        x3 = self.conv_layer(x2, self.encode3, self.batch_norm256, self.leaky_relu)
        x4 = self.conv_layer(x3, self.encode4, self.batch_norm512, self.leaky_relu)
        encoded = self.conv_layer(x4, self.encode5, self.batch_norm1024, self.leaky_relu)

        z = self.conv_layer(encoded, self.decode1, self.batch_norm512, self.leaky_relu, upsample=True) + x4
        z = self.conv_layer(z, self.decode2, self.batch_norm256, self.leaky_relu, upsample=True) + x3
        z = self.conv_layer(z, self.decode3, self.batch_norm128, self.leaky_relu, upsample=True) + edge2
        z = self.conv_layer(z, self.decode4, self.batch_norm64, self.leaky_relu, upsample=True) + edge1
        z = self.conv_layer(z, self.decode5, self.batch_norm64, self.leaky_relu, upsample=True) + edge_prelim

        z = self.conv_layer(z, self.classifier_conv, self.batch_norm2, self.leaky_relu, upsample=False)
        return z

class AttentionUpsampler(BaselineFCN):
    def __init__(self):
        super().__init__()
        self.attn_upsample1 = UpsamplerAttention(512, 1024, 1)
        self.attn_upsample2 = UpsamplerAttention(256, 512, 1)
        self.attn_upsample3 = UpsamplerAttention(128, 256, 1)
        self.attn_upsample4 = UpsamplerAttention(64, 128, 1)
        self.attn_upsample5 = UpsamplerAttention(32, 64, 1)
        self.classifier_conv = nn.Conv2d(32, 2, 3, 1, 1, bias=False)

    def forward(self, input):
        prelim = self.conv_layer(input, self.encode0, self.batch_norm32, self.leaky_relu)
        x1 = self.conv_layer(prelim, self.encode1, self.batch_norm64, self.leaky_relu)
        x2 = self.conv_layer(x1, self.encode2, self.batch_norm128, self.leaky_relu)
        x3 = self.conv_layer(x2, self.encode3, self.batch_norm256, self.leaky_relu)
        x4 = self.conv_layer(x3, self.encode4, self.batch_norm512, self.leaky_relu)
        encoded = self.conv_layer(x4, self.encode5, self.batch_norm1024, self.leaky_relu)

        z1 = self.attn_upsample1(contexts=[x4], mainstream=encoded)
        z2 = self.attn_upsample2(contexts=[x3], mainstream=z1)
        z3 = self.attn_upsample3(contexts=[x2], mainstream=z2)
        z4 = self.attn_upsample4(contexts=[x1], mainstream=z3)
        z5 = self.attn_upsample5(contexts=[prelim], mainstream = z4)

        z6 = self.conv_layer(z5, self.classifier_conv, self.batch_norm2, self.leaky_relu, upsample=False)
        return z6


class MultiRes(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias = True)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, 5, stride, 2, bias=True)
        self.conv7x7 = nn.Conv2d(in_channels, out_channels, 7, stride, 3, bias = True)

        self.batchnorm = nn.BatchNorm2d(out_channels, track_running_stats=True)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def conv_layer(self, input, conv, batch_norm, act):
        return batch_norm(act(conv(input)))

    def forward(self, input):
        conv3x3_out = self.conv_layer(input, self.conv3x3, self.batchnorm, self.leaky_relu)
        conv5x5_out = self.conv_layer(input, self.conv5x5, self.batchnorm, self.leaky_relu)
        conv7x7_out = self.conv_layer(input, self.conv7x7, self.batchnorm, self.leaky_relu)

        return conv3x3_out, conv5x5_out, conv7x7_out


class LocalAttentionUpsamplingModelMultiRec(BaselineFCN):
    def __init__(self):
        super().__init__()
        self.attn_upsample1 = UpsamplerLocalAttention(512, 512, 1)
        self.attn_upsample2 = UpsamplerLocalAttention(256, 512, 1)
        self.attn_upsample3 = UpsamplerLocalAttention(128, 256, 3)
        self.attn_upsample4 = UpsamplerLocalAttention(64, 128, 3)
        self.attn_upsample5 = UpsamplerLocalAttention(32, 64, 3)

        self.encode0 = MultiRes(1, 32, stride=1)
        self.encode1 = MultiRes(32, 64)
        self.encode2 = MultiRes(64, 128)
        self.encode3 = nn.Conv2d(128, 256, 3, 2, 1, bias=True) #MultiRes(128, 256)
        self.encode4 = nn.Conv2d(256, 512, 3, 2, 1, bias=True) #MultiRes(256, 512)
        self.encode5 = nn.Conv2d(512, 512, 3, 2, 1, bias=True)

        self.classifier_conv = nn.Conv2d(32, 2, 3, 1, 1, bias=True)

    def forward(self, input):
        prelim = self.encode0(input)        #self.conv_layer(input, self.encode0, self.batch_norm32, self.leaky_relu)
        x1 = self.encode1(prelim[0])           #self.conv_layer(prelim, self.encode1, self.batch_norm64, self.leaky_relu)
        x2 = self.encode2(x1[0])               #self.conv_layer(x1, self.encode2, self.batch_norm128, self.leaky_relu)
        x3 = self.conv_layer(x2[0], self.encode3, self.batch_norm256, self.leaky_relu)
        x4 = self.conv_layer(x3, self.encode4, self.batch_norm512, self.leaky_relu)
        encoded = self.conv_layer(x4, self.encode5, self.batch_norm512, self.leaky_relu)

        z1 = self.attn_upsample1(contexts=[x4], mainstream=encoded)
        z2 = self.attn_upsample2(contexts=[x3], mainstream=z1)
        z3 = self.attn_upsample3(contexts=x2, mainstream=z2)
        z4 = self.attn_upsample4(contexts=x1, mainstream=z3)
        z5 = self.attn_upsample5(contexts=prelim, mainstream = z4)

        z6 = self.conv_layer(z5, self.classifier_conv, self.batch_norm2, self.leaky_relu, upsample=False)
        return z6


class FCNEdgeDecoder(BaselineFCN):
    def __init__(self):
        super().__init__()
        self.encode5 = nn.Conv2d(512, 512, 3, 2, 1, bias=False)
        self.edge_decoder3 = DecoderAttentionParallel(enc_channels=32, dec_channels=32, num_contexts=1)
        self.classifier_conv = nn.Conv2d(32, 2, 3, 1, 1, bias = True)

        self.decode1 = nn.Conv2d(512, 512, 3, 1, 1, bias=True)
        self.decode5 = nn.Conv2d(64, 32, 3, 1, 1, bias=True)

        self.edge_prelim_conv = nn.Conv2d(1, 32, 3, 1, 1, bias=False)  # 32

    def forward(self, input):
        input, edge_map = input
        # pdb.set_trace()

        edge_prelim = self.conv_layer(edge_map, self.edge_prelim_conv, self.batch_norm32, self.leaky_relu)  # 256 x 256 x 32s
        # edge1 = self.conv_layer(edge_prelim, self.edge1_conv, self.batch_norm64, self.leaky_relu)  #  128 x 128 x 64
        # edge2 = self.conv_layer(edge1, self.edge2_conv, self.batch_norm128, self.leaky_relu)       # 64 x 64 x 128

        prelim = self.conv_layer(input, self.encode0, self.batch_norm32, self.leaky_relu)   # 256 x 256 x 32
        x1 = self.conv_layer(prelim, self.encode1, self.batch_norm64, self.leaky_relu)      # 128 x 128 x 64
        x2 = self.conv_layer(x1, self.encode2, self.batch_norm128, self.leaky_relu)         # 64 x 64 x 128
        x3 = self.conv_layer(x2, self.encode3, self.batch_norm256, self.leaky_relu)         # 32 x 32 x 256
        x4 = self.conv_layer(x3, self.encode4, self.batch_norm512, self.leaky_relu)         # 16 x 16 x 512
        encoded = self.conv_layer(x4, self.encode5, self.batch_norm512, self.leaky_relu)   # 8 x 8 x 1024

        z = self.conv_layer(encoded, self.decode1, self.batch_norm512, self.leaky_relu, upsample=True) + x4
        z = self.conv_layer(z, self.decode2, self.batch_norm256, self.leaky_relu, upsample=True) + x3
        z = self.conv_layer(z, self.decode3, self.batch_norm128, self.leaky_relu, upsample=True) + x2
        z = self.conv_layer(z, self.decode4, self.batch_norm64, self.leaky_relu, upsample=True) + x1
        z5 = self.conv_layer(z, self.decode5, self.batch_norm32, self.leaky_relu, upsample=True)
        decoded_features3 = self.edge_decoder3(contexts = [edge_prelim], decoded_features = z5) # 256 x 256 x 32

        # classifier layer
        z6 = self.conv_layer(decoded_features3, self.classifier_conv, self.batch_norm2, self.leaky_relu, upsample=False)
        return z6


class AttentionUpsamplerEdgeDecoder(AdditiveAttentionEdgeDecoder, AttentionUpsampler):
    def __init__(self):
        super().__init__()
        self.attn_upsample1 = UpsamplerAttentionParallel(512, 512, 1)
        self.attn_upsample2 = UpsamplerAttentionParallel(256, 512, 1)
        self.attn_upsample3 = UpsamplerAttentionParallel(128, 256, 1)
        self.attn_upsample4 = UpsamplerAttentionParallel(64, 128, 1)
        self.attn_upsample5 = UpsamplerAttentionParallel(32, 64, 1)

        self.encode5 = nn.Conv2d(512, 512, 3, 2, 1, bias=False)

        # self.edge_decoder1 = DecoderAttention(enc_channels=128, dec_channels=128, num_contexts=1)
        # self.edge_decoder2 = DecoderAttention(enc_channels=64, dec_channels=64, num_contexts=1)
        self.edge_decoder3 = DecoderAttentionParallel(enc_channels=32, dec_channels=32, num_contexts=1)

        self.classifier_conv = nn.Conv2d(32, 2, 3, 1, 1, bias = True)

        self.edge_prelim_conv = nn.Conv2d(1, 32, 3, 1, 1, bias=False)  # 32
        # self.edge1_conv = nn.Conv2d(32, 64, 3, 2, 1, bias=False)       # 64


    def forward(self, input):
        input, edge_map = input
        # pdb.set_trace()

        edge_prelim = self.conv_layer(edge_map, self.edge_prelim_conv, self.batch_norm32, self.leaky_relu)  # 256 x 256 x 32
        # edge1 = self.conv_layer(edge_prelim, self.edge1_conv, self.batch_norm64, self.leaky_relu)  #  128 x 128 x 64
        # edge2 = self.conv_layer(edge1, self.edge2_conv, self.batch_norm128, self.leaky_relu)       # 64 x 64 x 128

        prelim = self.conv_layer(input, self.encode0, self.batch_norm32, self.leaky_relu)   # 256 x 256 x 32
        x1 = self.conv_layer(prelim, self.encode1, self.batch_norm64, self.leaky_relu)      # 128 x 128 x 64
        x2 = self.conv_layer(x1, self.encode2, self.batch_norm128, self.leaky_relu)         # 64 x 64 x 128
        x3 = self.conv_layer(x2, self.encode3, self.batch_norm256, self.leaky_relu)         # 32 x 32 x 256
        x4 = self.conv_layer(x3, self.encode4, self.batch_norm512, self.leaky_relu)         # 16 x 16 x 512
        encoded = self.conv_layer(x4, self.encode5, self.batch_norm512, self.leaky_relu)   # 8 x 8 x 1024

        z1 = self.attn_upsample1(contexts=[x4], mainstream=encoded) # of shape 16 x 16 x 512
        z2 = self.attn_upsample2(contexts=[x3], mainstream=z1)   # is of shape 32 x 32 x 256
        z3 = self.attn_upsample3(contexts=[x2], mainstream=z2)  # is of shape 64 x 64 x 128
        # single step decoder attention
        # decoded_features1 = self.edge_decoder1(contexts = [edge2], decoded_features = z3)   # 64 x 64 x 128
        # upsample that
        z4 = self.attn_upsample4(contexts=[x1], mainstream = z3)   # 128 x 128 x 64
        # single step decoder attention
        # pdb.set_trace()
        # decoded_features2 = self.edge_decoder2(contexts = [edge1], decoded_features = z4)  # 128 x 128 x 64
        # upsample to 256
        z5 = self.attn_upsample5(contexts=[prelim], mainstream=z4)  # 256 x 256 x 32
        # decode at 256 scale
        decoded_features3 = self.edge_decoder3(contexts = [edge_prelim], decoded_features = z5) # 256 x 256 x 32

        # classifier layer
        z6 = self.conv_layer(decoded_features3, self.classifier_conv, self.batch_norm2, self.leaky_relu, upsample=False)
        return z6


class MultiContextualEdgeDecoder(BaselineFCN):
    def __init__(self):
        super().__init__()
        self.encode5 = nn.Conv2d(512, 512, 3, 2, 1, bias=False)
        self.edge_decoder3 = DecoderAttentionParallel(enc_channels=32, dec_channels=32, num_contexts=3)
        self.classifier_conv = nn.Conv2d(32, 2, 3, 1, 1, bias = True)

        self.decode1 = nn.Conv2d(512, 512, 3, 1, 1, bias=True)
        self.decode5 = nn.Conv2d(64, 32, 3, 1, 1, bias=True)

        self.edge_prelim_conv = MultiRes(32, 32, 1)  # 32

    def forward(self, input):
        input, edge_map = input
        # pdb.set_trace()

        edge_prelim = self.edge_prelim_conv(edge_map)  # 256 x 256 x 32s
        # edge1 = self.conv_layer(edge_prelim, self.edge1_conv, self.batch_norm64, self.leaky_relu)  #  128 x 128 x 64
        # edge2 = self.conv_layer(edge1, self.edge2_conv, self.batch_norm128, self.leaky_relu)       # 64 x 64 x 128

        prelim = self.conv_layer(input, self.encode0, self.batch_norm32, self.leaky_relu)   # 256 x 256 x 32
        x1 = self.conv_layer(prelim, self.encode1, self.batch_norm64, self.leaky_relu)      # 128 x 128 x 64
        x2 = self.conv_layer(x1, self.encode2, self.batch_norm128, self.leaky_relu)         # 64 x 64 x 128
        x3 = self.conv_layer(x2, self.encode3, self.batch_norm256, self.leaky_relu)         # 32 x 32 x 256
        x4 = self.conv_layer(x3, self.encode4, self.batch_norm512, self.leaky_relu)         # 16 x 16 x 512
        encoded = self.conv_layer(x4, self.encode5, self.batch_norm512, self.leaky_relu)   # 8 x 8 x 1024

        z = self.conv_layer(encoded, self.decode1, self.batch_norm512, self.leaky_relu, upsample=True) + x4
        z = self.conv_layer(z, self.decode2, self.batch_norm256, self.leaky_relu, upsample=True) + x3
        z = self.conv_layer(z, self.decode3, self.batch_norm128, self.leaky_relu, upsample=True) + x2
        z = self.conv_layer(z, self.decode4, self.batch_norm64, self.leaky_relu, upsample=True) + x1
        z5 = self.conv_layer(z, self.decode5, self.batch_norm32, self.leaky_relu, upsample=True)
        decoded_features3 = self.edge_decoder3(contexts = edge_prelim, decoded_features = z5) # 256 x 256 x 32

        # classifier layer
        z6 = self.conv_layer(decoded_features3, self.classifier_conv, self.batch_norm2, self.leaky_relu, upsample=False)
        return z6


class MultiResidualAttentionUpsamplerEdgeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_upsample1 = UpsamplerLocalAttention(512, 512, 1)
        self.attn_upsample2 = UpsamplerLocalAttention(256, 512, 1)
        self.attn_upsample3 = UpsamplerLocalAttention(128, 256, 1)
        self.attn_upsample4 = UpsamplerLocalAttention(64, 128, 1)
        self.attn_upsample5 = UpsamplerLocalAttention(32, 64, 1)

        # self.edge_decoder1 = DecoderAttentionParallel(enc_channels=128, dec_channels=128, num_contexts=3)
        # self.edge_decoder2 = DecoderAttentionParallel(enc_channels=64, dec_channels=64, num_contexts=1)
        # self.edge_decoder3 = DecoderAttentionParallel(enc_channels=32, dec_channels=32, num_contexts=1)

        self.classifier_conv = nn.Conv2d(32, 2, 3, 1, 1, bias = True)

        # self.edge_prelim_conv = nn.Conv2d(1, 32, 3, 1, 1, bias=True)     #MultiRes(1, 32, stride=1)
        # self.edge1_conv = MultiRes(32, 64)
        # self.edge2_conv = MultiRes(64, 128)

        self.edge_prelim_conv = MultiRes(1, 32, 1)

        self.encode0 = nn.Conv2d(1, 32, 3, 1, 1, bias=True) #MultiRes(1, 32, stride=1)
        self.encode1 = nn.Conv2d(32, 64, 3, 2, 1, bias=True) #MultiRes(32, 64)
        self.encode2 = nn.Conv2d(64, 128, 3, 2, 1, bias=True) #MultiRes(64, 128)
        self.encode3 = nn.Conv2d(128, 256, 3, 2, 1, bias=True)# MultiRes(128, 256)
        self.encode4 = nn.Conv2d(256, 512, 3, 2, 1, bias=True) #MultiRes(256, 512)
        self.encode5 = nn.Conv2d(512, 512, 3, 2, 1, bias=True)

        self.edge_decoder3 = DecoderAttentionParallel(enc_channels=32, dec_channels=32, num_contexts=3)

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        self.batch_norm2 = nn.BatchNorm2d(2, track_running_stats=True)
        self.batch_norm32 = nn.BatchNorm2d(32, track_running_stats=True)
        self.batch_norm64 = nn.BatchNorm2d(64, track_running_stats=True)
        self.batch_norm128 = nn.BatchNorm2d(128, track_running_stats=True)
        self.batch_norm256 = nn.BatchNorm2d(256, track_running_stats=True)
        self.batch_norm512 = nn.BatchNorm2d(512, track_running_stats=True)

    def forward(self, input):
        input, edge_map = input
        edge_prelim = self.edge_prelim_conv(edge_map)  # 256 x 256 x 32
        # edge1 = self.edge1_conv(edge_prelim[0])  #  128 x 128 x 64
        # edge2 = self.edge2_conv(edge1[0])       # 64 x 64 x 128

        prelim_3x3 = self.batch_norm32(self.leaky_relu(self.encode0(input)))   # 256 x 256 x 32       , prelim_5x5, prelim_7x7
        x1_3x3 = self.batch_norm64(self.leaky_relu(self.encode1(prelim_3x3)))      # 128 x 128 x 64     , x1_5x5, x1_7x7
        x2_3x3 = self.batch_norm128(self.leaky_relu(self.encode2(x1_3x3)))         # 64 x 64 x 128       , x2_5x5, x2_7x7
        x3_3x3 = self.batch_norm256(self.leaky_relu(self.encode3(x2_3x3)))         # 32 x 32 x 256   x3_5x5, x3_7x7
        x4 = self.batch_norm512(self.leaky_relu(self.encode4(x3_3x3)))  # 16 x 16 x 512
        encoded = self.batch_norm512(self.leaky_relu(self.encode5(x4)))   # 8 x 8 x 1024

        # pdb.set_trace()
        z1 = self.attn_upsample1(contexts=[x4], mainstream=encoded) # of shape 16 x 16 x 512
        z2 = self.attn_upsample2(contexts=[x3_3x3], mainstream=z1)   # is of shape 32 x 32 x 256      , x3_5x5, x3_7x7
        z3 = self.attn_upsample3(contexts=[x2_3x3], mainstream=z2)  # is of shape 64 x 64 x 128       , x2_5x5, x2_7x7
        # single step decoder attention
        # decoded_features1 = self.edge_decoder1(contexts = edge2, decoded_features = z3)   # 64 x 64 x 128
        # upsample that
        z4 = self.attn_upsample4(contexts=[x1_3x3], mainstream = z3)   # 128 x 128 x 64         , x1_5x5, x1_7x7
        # single step decoder attention
        # decoded_features2 = self.edge_decoder2(contexts = [edge1[0]], decoded_features = z4)  # 128 x 128 x 64
        # upsample to 256
        z5 = self.attn_upsample5(contexts=[prelim_3x3], mainstream=z4)  # 256 x 256 x 32       , prelim_5x5, prelim_7x7
        # pdb.set_trace()
        # decode at 256 scale
        decoded_features3 = self.edge_decoder3(contexts = edge_prelim, decoded_features = z5) # 256 x 256 x 32

        # classifier layer
        z6 = self.batch_norm2(self.leaky_relu(self.classifier_conv(decoded_features3)))
        return z6

def ensure_exists(paths):
    for path in paths:
        Path.mkdir(path, exist_ok=True, parents=True)

def obtain_latest_checkpoint(dir_path):
    files = sorted([(path, path.name) for path in dir_path.iterdir()], key = lambda x: int(re.search('\d+', x[1]).group()), reverse = True)
    return files[0][0] if len(files) else None

def load_training_stats(path):
    return json.load(open(path, 'r'))

def dice_dissim(input, target):
    smooth = 0.1

    iflat = input.reshape(-1)
    tflat = target.reshape(-1)
    intersection = (iflat * tflat).sum()

    dice_coeff =  ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))
    return 1 - dice_coeff

def model_train(dataloader_train, dataloader_val, model, model_name, epochs = 50, path_to_save= Path('ct_outputs/'), edges = False, deeplab_special = False):
    # move model to device, so that optimizer can update cuda parameters
    model = model.to(device)

    # Setup Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))
    lr_decay = None
    # lr_decay = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=6)
    # lr_decay = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    model_path_to_save = path_to_save / 'ct_models' / model_name / 'epoch%d.pth'
    metrics_path_to_save = path_to_save / 'ct_metrics' / model_name / 'ct_metrics.json'

    # makes sure these directories are created.
    ensure_exists([model_path_to_save.parent, metrics_path_to_save.parent])

    start_epoch = 1
    best_epoch_dice = 0.0
    training_stats = {}

    latest_checkpoint_file = obtain_latest_checkpoint(model_path_to_save.parent)
    print('Loaded ', latest_checkpoint_file, '!')
    if latest_checkpoint_file is not None:
        checkpoint = torch.load(str(latest_checkpoint_file))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['opt_state_dict'])
        # pdb.set_trace()
        # if 'lr_decay' in checkpoint:
            # lr_decay.load_state_dict(checkpoint['lr_decay'])
        start_epoch = checkpoint['epoch']
        best_epoch_dice = checkpoint['best_epoch_dice']
        training_stats = load_training_stats(str(metrics_path_to_save))

    if (device.type == 'cuda') and (ngpu > 1):
        print("Let's use ", torch.cuda.device_count(), " GPUs for the Model!")
        model = nn.DataParallel(model, list(range(ngpu)))

    # Define losses
    entropy_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9]).to(device), reduction='mean')

    # epochs
    try:
        for epoch in range(start_epoch, start_epoch + epochs):
            epoch_loss = epoch_acc = epoch_dice = 0
            model.train()
            for batch_index, data in enumerate(dataloader_train):
                cts = data[0].to(device)
                masks = data[1].to(device)
                edge_map = None
                if edges:
                    edge_map = data[2].to(device)
                model.zero_grad()

                activated_output = model((cts, edge_map)) if edge_map is not None else model(cts)
                if deeplab_special:
                    activated_output = activated_output['out']
                softmax_activated_output = nn.functional.softmax(activated_output, dim=1)
                # DICE
                dice_loss = dice_dissim(input = softmax_activated_output[:, 1, :, :], target=masks)
                # CROSS ENTROPY
                log_loss = entropy_loss(input=activated_output, target= masks.long()) # if epoch>5 else 0.0
                # TOTAL ERROR
                error = dice_loss + log_loss

                # pdb.set_trace()
                # acc gradients
                error.backward()
                # perform grad update
                optimizer.step()

                # model accuracy
                _, model_output_indices = torch.max(softmax_activated_output, dim = 1)

                model_acc = torch.sum(model_output_indices == masks.long()).float() / torch.numel(model_output_indices)

                # Training Stats:
                # '''
                print('[%d/%d] [%d/%d]\tTRAIN Dice: %.4f\tError: %.4f\tAccuracy: %.4f' % (epoch,
                                                                             start_epoch + epochs,
                                                                             batch_index+1,
                                                                             len(dataloader_train),
                                                                             (1-dice_loss).item(),
                                                                              log_loss.item(),
                                                                              model_acc.item()))
                # '''
                epoch_loss += log_loss.item(); epoch_acc += model_acc.item(); epoch_dice += (1-dice_loss).item()
                # del cts, masks, edge_map, error, model_acc, dice_loss, log_loss, activated_output, softmax_activated_output
                # torch.cuda.empty_cache()

            with torch.no_grad():
                val_acc = val_loss = val_dice = 0.0
                # model.eval()
                for data in dataloader_val:
                    ct, mask = data[0].to(device), data[1].to(device)
                    val_edge_map = None
                    if len(data)==3:
                        val_edge_map = data[2].to(device)
                    output = model((ct, val_edge_map)) if val_edge_map is not None else model(ct)
                    if deeplab_special:
                        output = output['out']
                    softmax_output = nn.functional.softmax(output, dim=1)
                    _, pred_classes = torch.max(softmax_output, dim=1)
                    dice_val = 1 - dice_dissim(softmax_output[:, 1, :, :], mask).cpu()
                    val_acc += (torch.sum(pred_classes == mask).float() / torch.numel(pred_classes)).cpu().item()
                    val_loss += entropy_loss(input=output, target=mask.long()).cpu().item() # if epoch>5 else 0.0
                    val_dice += dice_val.item()

            if epoch_dice > best_epoch_dice:
                best_epoch_dice = epoch_dice
                checkpoint_model(epoch, model, optimizer, lr_decay, best_epoch_dice, str(model_path_to_save) % epoch)

            epoch_stat = {
                'loss': epoch_loss / len(dataloader_train),
                'acc': epoch_acc/len(dataloader_train),
                'dice': epoch_dice / len(dataloader_train),
                'val_dice': val_dice/ len(dataloader_val),
                'val_loss': val_loss/len(dataloader_val),
                'val_acc': val_acc/len(dataloader_val)
            }

            print('\nEpoch [%d/%d]\tTRAIN\tDICE: %.4f\tLoss:%.4f\tAccuracy:%.4f\tLR:%.4f' % (epoch,
                                                                                             start_epoch + epochs,
                                                                                             epoch_stat['dice'],
                                                                                             epoch_stat['loss'],
                                                                                             epoch_stat['acc'],
                                                                                             optimizer.state_dict()['param_groups'][0]['lr']))
            print('Epoch [%d/%d]\tVAL\tDICE:%.4f\tLoss:%.4f\tAccuracy:%.4f\n' % (epoch, start_epoch + epochs, epoch_stat['val_dice'], epoch_stat['val_loss'], epoch_stat['val_acc']))
            # save stat to global stats.
            training_stats['epoch%d' % epoch] = epoch_stat
            # learning rate scheduler
            # lr_decay.step(epoch_stat['dice'])
    except KeyboardInterrupt:
        print('Oh its an interrupt.')
    finally:
        print('Saving checkpoints and metrics')
        checkpoint_metrics(training_stats, metrics_path= str(metrics_path_to_save))
        # if epoch_stat['val_dice'] > best_val_dice:
        checkpoint_model(epoch, model, optimizer, lr_decay, best_epoch_dice, str(model_path_to_save) % epoch)
        print('DONE saving')

def checkpoint_metrics(object, metrics_path):
    json.dump(object, open(metrics_path, 'w'), indent=2)

def checkpoint_model(epoch, model, opt, lr_decay, best_epoch_dice, model_path):
    model_state_dict = model.module.state_dict() if (device.type == 'cuda') else model.state_dict()
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'opt_state_dict': opt.state_dict(),
        'best_epoch_dice': best_epoch_dice,
        'lr_decay': lr_decay.state_dict() if lr_decay !=None else None
    }, model_path)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__2':
    # upattn_parallel = UpsamplerAttentionParallel(32, 64, 3).to(device)
    # upattn = UpsamplerAttention(32, 64, 3).to(device)
    # contexts = [torch.randn(32, 32, 256, 256).to(device)] * 3
    # features = torch.randn(32, 64, 128, 128).to(device)

    decoder_attn_parallel = DecoderAttentionParallel(32, 64, 3).to(device)
    contexts = [torch.randn(10, 32, 256, 256).to(device)] * 3
    features = torch.randn(10, 64, 256, 256).to(device)

    start = time.time()
    out = decoder_attn_parallel(contexts, features)
    end = time.time()
    print('parallel: ', (end-start), ' seconds')

    device1 = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    decoder_attn = DecoderAttention(32, 64, 3).to(device1)
    contexts = [torch.randn(10, 32, 256, 256).to(device1)] * 3
    features = torch.randn(10, 64, 256, 256).to(device1)
    start = time.time()
    out = decoder_attn(contexts, features)
    end = time.time()
    print('traversal: ', (end-start), ' seconds')



if __name__ == '__main__':
    # attn = DecoderAttention(512, 1024, 1)
    # contexts = [torch.randn(32, 512, 16, 16)] * 1
    # dec = torch.randn(32, 1024, 16, 16)
    # f = attn(contexts, dec)
    # print(f.shape)

    # upattn = UpsamplerAttention(512, 1024, 3)
    # contexts = [torch.randn(32, 512, 16, 16)] * 3
    # features = torch.randn(32, 1024, 8, 8)
    # out = upattn(contexts, features)
    # print(out.shape)

    x = torch.randn(10, 1, 256, 256)
    edge_map = torch.randn(10, 1, 256, 256)

    baseline_fcn = BaselineFCN()
    additive_attn_fcn = AdditiveAttentionFCN()
    additive_attn_edge_decoder = AdditiveAttentionEdgeDecoder()
    attn_upsampler = AttentionUpsampler()
    attn_upsampler_edge_decoder = AttentionUpsamplerEdgeDecoder()
    multi_residual_attn_upsampler_edge_decoder = MultiResidualAttentionUpsamplerEdgeDecoder()
    fcn_edge_decoder = FCNEdgeDecoder()
    local_attn_upsampling_multi_rec = LocalAttentionUpsamplingModelMultiRec()

    # lnet = Linknet().to(device)           # OKAY
    # r2u_net = R2U_Net().to(device)
    # fcn_edge_decoder = fcn_edge_decoder.to(device)    # 39M - OKAY
    # pspnet = PSPNet().to(device)          # 21M - OKAY
    # fcn = FCN_ResNet50().to(device)

    model = fcn_edge_decoder #multi_residual_attn_upsampler_edge_decoder

    # print('baseline: ' , baseline_fcn(x).shape)
    # print('additive fcn: ', additive_attn_fcn(x).shape)
    # print('additive attn edge decoder: ', additive_attn_edge_decoder((x, edge_map)).shape)
    # print('attn upsampler: ', attn_upsampler(x).shape)
    # print('attn upsampler edge decoder: ', attn_upsampler_edge_decoder((x, edge_map)).shape)
    # print('#parameters: ', count_parameters(attn_upsampler_edge_decoder))

    transforms = torchvision.transforms.Compose([
        HoriFlip(0.5),
        Affine(translate_xy=(0.1, 0.1), shear_angle_range=(-5, 5), rotate_angle_range=(-10, 10))
    ])

    split_data = train_val_splits(dataset_path='processed_data_proper')

    ct_seg_dataset_train = COVID19_CT_dataset(samples=split_data['train'], scan_norm=(0.5330, 0.3477),
                                              transforms=transforms, edges = True)
    ct_seg_dataset_val = COVID19_CT_dataset(samples=split_data['val'], scan_norm=(0.5330, 0.3477),
                                            transforms=None, edges = True)
    ct_seg_dataset_test = COVID19_CT_dataset(samples=split_data['test'], scan_norm=(0.5330, 0.3477),
                                            transforms=None, edges=True)

    ct_seg_dataloader_train = torch.utils.data.DataLoader(ct_seg_dataset_train, batch_size=40, num_workers=50)   # fcn edge decoder 40
    ct_seg_dataloader_val = torch.utils.data.DataLoader(ct_seg_dataset_val, batch_size=40, num_workers=50)
    ct_seg_dataloader_test = torch.utils.data.DataLoader(ct_seg_dataset_test, batch_size=40, num_workers=50)

    # for i, data in enumerate(ct_seg_dataloader_val):
        # ct, mask = data
        # print(i, 'TRAIN data: ', ct.shape, mask.shape)

    # print('train/val sizes:', len(ct_seg_dataset_train), len(ct_seg_dataset_val))
    # exit(0)
    # for name, named_parameter in multi_residual_attn_upsampler_edge_decoder.named_parameters():
    #     print(name,': ', named_parameter.numel())

    print('param: ', count_parameters(model))

    # for name, child in multi_residual_attn_upsampler_edge_decoder.named_children():
    #     print(name, count_parameters(child))

    # x = torch.randn(10, 1, 256, 256)
    # edges = torch.randn(10, 1, 256, 256)
    # start = time.time()
    # pdb.set_trace()
    # out = multi_residual_attn_upsampler_edge_decoder((x,edges))
    # end = time.time()
    # print('time: ', (end-start), ', out:', out.shape)

    # x = torch.randn(10, 1, 256, 256)
    # m = AdditiveAttentionFCN()
    # print(m(x).shape)
    # summary(m, (1, 256, 256))

    model_train(ct_seg_dataloader_train, ct_seg_dataloader_val, model, 'FCNEdgeDecoder', epochs=50, path_to_save=Path('ct_outputs'), edges = True, deeplab_special= False)