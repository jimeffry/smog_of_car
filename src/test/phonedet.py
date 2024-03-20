###############################################
#created by :  lixiaoyu80
#Time:  2021/3/23 10:09
#project: mobilephone recognition
#rversion: 0.1
#tool:   python 3.6
#modified:
#description  
####################################################
import sys
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

# import cv2
import time
import numpy as np
# from tqdm import tqdm
# from matplotlib import pyplot as plt
from config import *

class MobilePhoneDet(object):
    '''
    require constants: PhoneDetModelPath, DetImgHeight,DetImgWidth
    '''
    def __init__(self,gpu_id=-1):
        if gpu_id >=0 and torch.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False
        if self.use_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
        if gpu_id >=0 :
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        self.loadmodel(PhoneDetModelPath)

    def loadmodel(self,modelpath):
        if self.use_cuda:
            device = 'cuda'
        else:
            device = 'cpu'
        self.net = torch.jit.load(modelpath).to(device)
        self.net.eval()
        # if self.use_cuda:
            # cudnn.benckmark = True

    def resize_scale(self,image, target_size):
        '''
        image: numpy_array
        target_size: img_w,img_h
        '''
        ih, iw = target_size
        # 原始图片尺寸
        h,  w, _ = image.shape
        # 计算缩放后图片尺寸
        scale = min(iw/w, ih/h)
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))
        # 创建一张画布，画布的尺寸就是目标尺寸 fill_value=128为灰色画布
        image_paded = np.full(shape=[ih, iw, 3], fill_value=128)
        dw, dh = (iw - nw) // 2, (ih-nh) // 2
        # 将缩放后的图片放在画布中央
        image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
        return image_paded.astype(np.uint8)

    def preprocess(self,imgs):
        # rgb_mean = np.array([123.,117.,104.])[np.newaxis, np.newaxis,:].astype('float32')
        rgb_mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis,:].astype('float32')
        rgb_std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis,:].astype('float32')
        img_out = []
        for img in imgs:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            w,h = img.shape[:2]
            if w != DetImgWidth or h != DetImgHeight:
                # img = cv2.resize(img,(cfg.InputSize_w,cfg.InputSize_h))
                img = self.resize_scale(img,(DetImgWidth,DetImgHeight))
            img = img.astype('float32')
            img /= 255.0
            img -= rgb_mean
            img /= rgb_std
            img = np.transpose(img,(2,0,1))
            img_out.append(img)
        return np.array(img_out)

    def inference(self,imglist):
        # t1 = time.time()
        imgs = self.preprocess(imglist)
        bt_img = torch.from_numpy(imgs)
        if self.use_cuda:
            bt_img = bt_img.cuda()
        output = self.net(bt_img)
        # output = F.softmax(output,dim=-1)
        pred_cls = torch.argmax(output,dim=1)
        # t2 = time.time()
        # print('consuming:',t2-t1)
        return output.data.cpu().numpy(),pred_cls.data.cpu().numpy()