#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image as RosImage

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils import data

#from FCHarDNet.pytorch_bn_fusion.bn_fusion import fuse_bn_recursively
#from FCHarDNet.ptsemseg.models import get_model
#from FCHarDNet.ptsemseg.loader.cityscapes_loader import decode_segmap
#from FCHarDNet.ptsemseg.utils import convert_state_dict

import cv2
from matplotlib import gridspec
from matplotlib import pyplot as plt
from PIL import Image
from cv_bridge import CvBridge, CvBridgeError
from collections import OrderedDict

torch.backends.cudnn.benchmark = True

## From https://github.com/PingoLH/FCHarDNet/ptsemseg/models/hardnet.py ##

class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                                          stride=stride, padding=kernel//2, bias = False))
        self.add_module('norm', nn.BatchNorm2d(out_channels))
        self.add_module('relu', nn.ReLU(inplace=True))

        #print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)

    def forward(self, x):
        return super().forward(x)
        


class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
          return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
          dv = 2 ** i
          if layer % dv == 0:
            k = layer - dv
            link.append(k)
            if i > 0:
                out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
          ch,_,_ = self.get_link(i, base_ch, growth_rate, grmul)
          in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels
 
    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0 # if upsample else in_channels
        for i in range(n_layers):
          outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
          self.links.append(link)
          use_relu = residual_out
          layers_.append(ConvLayer(inch, outch))
          if (i % 2 == 0) or (i == n_layers - 1):
            self.out_channels += outch
        #print("Blk out =",self.out_channels)
        self.layers = nn.ModuleList(layers_)


    def forward(self, x):
        layers_ = [x]
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)
        t = len(layers_)
        out_ = []
        for i in range(t):
          if (i == 0 and self.keepBase) or \
             (i == t-1) or (i%2 == 1):
              out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out

class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #print("upsample",in_channels, out_channels)

    def forward(self, x, skip, concat=True):
        out = F.interpolate(
                x,
                size=(skip.size(2), skip.size(3)),
                mode="bilinear",
                align_corners=True,
                            )
        if concat:                            
          out = torch.cat([out, skip], 1)
          
        return out

class hardnet(nn.Module):
    def __init__(self, n_classes=19):
        super(hardnet, self).__init__()

        first_ch  = [16,24,32,48]
        ch_list = [  64, 96, 160, 224, 320]
        grmul = 1.7
        gr       = [  10,16,18,24,32]
        n_layers = [   4, 4, 8, 8, 8]

        blks = len(n_layers) 
        self.shortcut_layers = []

        self.base = nn.ModuleList([])
        self.base.append (
             ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3,
                       stride=2) )
        self.base.append ( ConvLayer(first_ch[0], first_ch[1],  kernel=3) )
        self.base.append ( ConvLayer(first_ch[1], first_ch[2],  kernel=3, stride=2) )
        self.base.append ( ConvLayer(first_ch[2], first_ch[3],  kernel=3) )

        skip_connection_channel_counts = []
        ch = first_ch[3]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i])
            ch = blk.get_out_ch()
            skip_connection_channel_counts.append(ch)
            self.base.append ( blk )
            if i < blks-1:
              self.shortcut_layers.append(len(self.base)-1)

            self.base.append ( ConvLayer(ch, ch_list[i], kernel=1) )
            ch = ch_list[i]
            
            if i < blks-1:            
              self.base.append ( nn.AvgPool2d(kernel_size=2, stride=2) )


        cur_channels_count = ch
        prev_block_channels = ch
        n_blocks = blks-1
        self.n_blocks =  n_blocks

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        self.conv1x1_up    = nn.ModuleList([])
        
        for i in range(n_blocks-1,-1,-1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            self.conv1x1_up.append(ConvLayer(cur_channels_count, cur_channels_count//2, kernel=1))
            cur_channels_count = cur_channels_count//2

            blk = HarDBlock(cur_channels_count, gr[i], grmul, n_layers[i])
            
            self.denseBlocksUp.append(blk)
            prev_block_channels = blk.get_out_ch()
            cur_channels_count = prev_block_channels


        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
               out_channels=n_classes, kernel_size=1, stride=1,
               padding=0, bias=True)
               

    def forward(self, x):
        
        skip_connections = []
        size_in = x.size()
        
        
        for i in range(len(self.base)):
            x = self.base[i](x)
            if i in self.shortcut_layers:
                skip_connections.append(x)
        out = x
        
        for i in range(self.n_blocks):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip, True)
            out = self.conv1x1_up[i](out)
            out = self.denseBlocksUp[i](out)
        
        out = self.finalConv(out)
        
        out = F.interpolate(
                            out,
                            size=(size_in[2], size_in[3]),
                            mode="bilinear",
                            align_corners=True)
        return out
## END ##

## From https://github.com/PingoLH/FCHarDNet/ptsemseg/loader/cityscapes_loader/ ##

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

## End ##

## From https://github.com/MIPT-Oulu/pytorch_bn_fusion/ ##

def fuse_bn_sequential(block):
    """
    This function takes a sequential block and fuses the batch normalization with convolution

    :param model: nn.Sequential. Source resnet model
    :return: nn.Sequential. Converted block
    """
    if not isinstance(block, nn.Sequential):
        return block
    stack = []
    for m in block.children():
        if isinstance(m, nn.BatchNorm2d):
            if isinstance(stack[-1], nn.Conv2d):
                bn_st_dict = m.state_dict()
                conv_st_dict = stack[-1].state_dict()

                # BatchNorm params
                eps = m.eps
                mu = bn_st_dict['running_mean']
                var = bn_st_dict['running_var']
                gamma = bn_st_dict['weight']

                if 'bias' in bn_st_dict:
                    beta = bn_st_dict['bias']
                else:
                    beta = torch.zeros(gamma.size(0)).float().to(gamma.device)

                # Conv params
                W = conv_st_dict['weight']
                if 'bias' in conv_st_dict:
                    bias = conv_st_dict['bias']
                else:
                    bias = torch.zeros(W.size(0)).float().to(gamma.device)

                denom = torch.sqrt(var + eps)
                b = beta - gamma.mul(mu).div(denom)
                A = gamma.div(denom)
                bias *= A
                A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

                W.mul_(A)
                bias.add_(b)

                stack[-1].weight.data.copy_(W)
                if stack[-1].bias is None:
                    stack[-1].bias = torch.nn.Parameter(bias)
                else:
                    stack[-1].bias.data.copy_(bias)

        else:
            stack.append(m)

    if len(stack) > 1:
        return nn.Sequential(*stack)
    else:
        return stack[0]


def fuse_bn_recursively(model):
    for module_name in model._modules:
        model._modules[module_name] = fuse_bn_sequential(model._modules[module_name])
        if len(model._modules[module_name]._modules) > 0:
            fuse_bn_recursively(model._modules[module_name])

    return model
## End ##

class Semantic_segmentation:
    def __init__(self):
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", RosImage, self.image_callback, queue_size=1)
        self.segmented_image_pub = rospy.Publisher("/fchardnet/segmented_image", RosImage, queue_size=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.use_subscribed_images_stamp = True
        self.colors = [  # [  0,   0,   0],
                        [128, 64, 128],
                        [244, 35, 232],
                        [70, 70, 70],
                        [102, 102, 156],
                        [190, 153, 153],
                        [153, 153, 153],
                        [250, 170, 30],
                        [220, 220, 0],
                        [107, 142, 35],
                        [152, 251, 152],
                        [0, 130, 180],
                        [220, 20, 60],
                        [255, 0, 0],
                        [0, 0, 142],
                        [0, 0, 70],
                        [0, 60, 100],
                        [0, 80, 100],
                        [0, 0, 230],
                        [119, 11, 32],
                    ]

        rospy.init_node('semantic_segmentation', anonymous=True)
        self.use_subscribed_images_stamp = rospy.get_param("/recognition/segmentation_publisher/USE_SUBSCRIBED_IMAGES_STAMP")
        self.model_name = rospy.get_param("/recognition/segmentation_publisher/MODEL_NAME")
        self.num_classes = rospy.get_param("/recognition/segmentation_publisher/NUM_CLASSES")
        self.model_path = rospy.get_param("/recognition/segmentation_publisher/MODEL_PATH")
        self.bn_fusion = rospy.get_param("/recognition/segmentation_publisher/BN_FUSION")
        self.update_bn = rospy.get_param("/recognition/segmentation_publisher/UPDATE_BN")
        self.input_size = rospy.get_param("/recognition/segmentation_publisher/INPUT_SIZE")
        self.label_colours = dict(zip(range(self.num_classes), self.colors))

        self.model = hardnet(self.num_classes).to(self.device)
        self.state = convert_state_dict(torch.load(self.model_path)["model_state"])
        self.model.load_state_dict(self.state)
        
        if self.bn_fusion:
            self.model = fuse_bn_recursively(self.model)
            print(self.model)

        if self.update_bn:
            print("Reset BatchNorm and recalculate mean/var")
            self.model.apply(reset_batchnorm)
            self.model.train()
        else:
            self.model.eval()
        
        self.model.to(self.device)

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.num_classes):
            r[temp == l] = self.abel_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]
        
        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        
        return rgb

    def image_callback(self, img):
        try:
            cv_image = CvBridge().imgmsg_to_cv2(img, "bgr8")
            pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            width, height = pil_image.size
            resize_ratio = 1.0 * self.input_size / max(width, height)
            target_size = (int(resize_ratio * width), int(resize_ratio * height))
            resized_image = pil_image.convert('RGB').resize(target_size, Image.ANTIALIAS)
            input_image = resized_image.to(self.device)
            
            with torch.no_grad():
                output = self.model(input_image)
            
            seg_image = np.squeeze(output.data.max(1)[1].cpu().numpy(), axis=0)
            seg_image = decode_segmap(self.num_classes, seg_image)
            cv_seg_image = np.asarray(seg_image)
            pub_seg_image = CvBridge().cv2_to_imgmsg(cv_seg_image, "rgb8")
            pub_seg_image.header = img.header
            if not self.use_subscribed_images_stamp:
                pub_seg_image.header.stamp = rospy.get_rostime()
            self.segmented_image_pub.publish(pub_seg_image)

        except CvBridgeError as e:
            print(e)

    def process(self):
        #rospy.init_node('semantic_segmentation', anonymous=True)
        rospy.spin()

if __name__ == "__main__":
    ss = Semantic_segmentation()

    try:
        ss.process()

    except rospy.ROSInterruptException: pass_

