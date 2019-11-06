#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image as RosImage

import torch
import torch.nn
import torch.nn.functional as F
import numpy as np

from torch.utils import data

from FCHarDNet.pytorch_bn_fusion.bn_fusion import fuse_bn_recursively
from FCHarDNet.ptsemseg.models import get_model
from FCHarDNet.ptsemseg.loader import get_loader
from FCHarDNet.ptsemseg.loader.cityscapes_loader import decode_segmap
from FCHarDNet.ptsemseg.metrics import runningScore
from FCHarDNet.ptsemseg.utils import convert_state_dict

import cv2
from matplotlib import gridspec
from matplotlib import pyplot as plt
from PIL import Image
from cv_bridge import CvBridge, CvBridgeError

torch.backends.cudnn.benchmark = True


class Semantic_segmentation:
    def __init__(self):
        self.self.image_sub = rospy.Subscriber("/usb_cam/image_raw", RosImage, self.image_callback, queue_size=1)
        self.segmented_image_pub = rospy.Publisher("/fchardnet/segmented_image", RosImage, queue_size=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.use_subscribed_images_stamp = True
        rospy.init_node('semantic_segmentation', anonymous=True)
        if rospy.has_param("USE_SUBSCRIBED_IMAGES_STAMP"):
            self.use_subscribed_images_stamp = rospy.get_param("USE_SUBSCRIBED_IMAGES_STAMP")
        if rospy.has_param("MODEL_NAME"):
            self.model_name = rospy.get_param("MODEL_NAME")
        if rospy.has_param("NUM_CLASSES"):
            self.num_classes = rospy.get_param("NUM_CLASSES")
        if rospy.has_param("MODEL_PATH"):
            self.model_path = rospy.get_param("MODEL_PATH")
        if rospy.has_param("BN_FUSION"):
            self.bn_fusion = rospy.get_param("BN_FUSION")
        if rospy.has_param("UPDATE_BN"):
            self.update_bn = rospy.get_param("UPDATE_BN")
        if rospy.has_param("INPUT_SIZE"):
            self.input_size = rospy.get_param("INPUT_SIZE")

        self.model = get_model(self.model_name, self.num_classes).to(self.device)
        self.state = convert_state_dict(torch.load(self.model_path)["model_state"])
        self.model.load_state_dict(self.state)
        
        if self.bn_fusion:
            self.model = fuse_bn_recursively(self.model)
            print(model)

        if self.update_bn:
            print("Reset BatchNorm and recalculate mean/var")
            self.model.apply(reset_batchnorm)
            self.model.train()
        else:
            self.model.eval()
        
        self.model.to(self.device)

    def image_callback(self, img):
        try:
            cv_image = CvBridge().imgmsg_to_cv2(img, "bgr8")
            pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            width, height = image.size
            resize_ratio = 1.0 * self.input_size / max(width, height)
            target_size = (int(resize_ratio * width), int(resize_ratio * height))
            resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
            input_image = resized_image.to(self.device)
            
            with torch.no_grad():
                output = model(input_image)
            
            seg_image = np.squeeze(output.data.max(1)[1].cpu().numpy(), axis=0)
            seg_image = decode_segmap(seg_image)
            cv_seg_image = np.asarray(seg_image)
            pub_seg_image = CvBridge().cv2_to_imgmsg(seg_image, "rgb8")
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

