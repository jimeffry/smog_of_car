import sys
import os
from collections import defaultdict
import torch
import torch.utils.data as data
import cv2
import numpy as np
import random 
import csv
import math
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfg

class ReadDataset(data.Dataset):
    """VOC Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """
    def __init__(self, root,annotxt,csvfile=None,train_fg=True):
        self.imgdir = root
        self.train_fg = train_fg
        self.annotations = []
        self.loadtxt(annotxt)
        self.total_num = self.__len__()
        self.shulf_num = list(range(self.total_num))
        random.shuffle(self.shulf_num)
        # self.rgb_mean = np.array([123.,117.,104.])[np.newaxis, np.newaxis,:].astype('float32')
        self.rgb_mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis,:].astype('float32')
        self.rgb_std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis,:].astype('float32')
        # self.rgb_mean = np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis,:].astype('float32')
        # self.rgb_std = np.array([0.225, 0.225, 0.225])[np.newaxis, np.newaxis,:].astype('float32')
        if csvfile is not None:
            self.soft_label_dict = self.readcsv(csvfile,cfg.FaceProperty)
        else:
            self.soft_label_dict = None

    def loadtxt(self,annotxt):
        self.data_r = open(annotxt,'r')
        voc_annotations = self.data_r.readlines()
        for tmp in voc_annotations:
            tmp_splits = tmp.strip().split(',')
            img_path = os.path.join(self.imgdir,tmp_splits[0])
            # img_name = tmp_splits[0].split('/')[-1][:-4] if len(tmp_splits[0].split('/')) >0 else tmp_splits[0][:-4]
            label = int(tmp_splits[1])
            labels = [img_path,label]
            self.annotations.append(labels)

    def __getitem__(self, index):
        im, gt = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.annotations)

    def pull_item(self, index):
        idx = self.shulf_num[index]
        tmp_annotation = self.annotations[idx]
        tmp_path = tmp_annotation[0]
        img_data = cv2.imread(tmp_path)
        gt_label = tmp_annotation[1]
        if self.soft_label_dict is not None:
            key_name = tmp_path.split('/')[-1]
            t_label = np.array(self.soft_label_dict[key_name])
            # h_label = np.zeros(cfg.CLS_NUM)
            # h_label[int(gt_label)]=1
            # gt_label = h_label *0.7 + t_label *0.3
            if t_label[1] < 0.8:
                tmp_s = t_label[1]
                t_label[1] = t_label[0]
                t_label[0] = tmp_s
            gt_label = t_label
        img = cv2.cvtColor(img_data,cv2.COLOR_BGR2RGB)
        img = self.prepro(img)
        # return torch.from_numpy(img).permute(2,0,1), torch.from_numpy(gt_label)
        return torch.from_numpy(img).permute(2,0,1), gt_label

    def prepro(self,img):
        if self.train_fg:
            img = self.mirror(img)
            if random.randrange(2):
               img = self.sp_noise(img)
            if random.randrange(2):
               img = self.gasuss_noise(img)
            if random.randrange(2):
                img = self.Brightness(img)
            if random.randrange(2):
                img = self.Distort(img)
        img = self.resize_subtract_mean(img)
        # img,gt = transform_crop(img,gt)
        return img

    def mirror(self,image):
        if random.randrange(2):
            image = image[:, ::-1,:]
        if random.randrange(2):
            image = image[::-1, :,:]
        return image

    def resize_subtract_mean(self,image):
        h,w,_ = image.shape
        if h != cfg.InputSize_h or w != cfg.InputSize_w:
            image = cv2.resize(image,(int(cfg.InputSize_w),int(cfg.InputSize_h)))
        image = image.astype(np.float32)
        image = image / 255.0
        image -= self.rgb_mean
        image = image / self.rgb_std
        return image

    def readcsv(self,filein,key_list):
        '''
        filein: result.csv  ---image_name,cls_num1_name,cls_num1_fg, ...
        ---em key_list = ['front','back','background']
        return: data_dict
        '''
        data_dict = defaultdict(list)
        fr = open(filein,'r')
        reader = csv.DictReader(fr)
        for tmp_item in reader:
            imgpath = tmp_item['filename']
            img_spl = imgpath.strip().split('/')
            if len(img_spl)>1:
                key_name = img_spl[-1]
            else:
                key_name = imgpath.strip()
            for tmp in key_list:
                data_dict[key_name].append(float(tmp_item[tmp]))
        fr.close()
        return data_dict
    
    def sp_noise(self,image,prob=0.0001):
        '''
        Salt and pepper noise
        prob: noise ratio
        '''
        output = np.zeros(image.shape,np.uint8)
        thres = 1 - prob
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        return output
    
    def gasuss_noise(self,image, mean=0, var=0.0001):
        '''
        add gasuss noise
        mean 
        var 
        '''
        image = np.array(image/255, dtype=float)
        noise = np.random.normal(mean, var ** 0.5, image.shape)
        out = image + noise
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(out, low_clip, 1.0)
        out = np.uint8(out*255)
        return out

    def Rotate(self, img,degree=45,size=0.7):
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2),degree, size)
        return cv2.warpAffine(img, M, (cols, rows))

    def Brightness(self,img,factor=0.5):
        return np.uint8(np.clip((factor * img + 125*(1-factor)), 0, 255))
    
    def Distort(self,image):
        def _convert(image, alpha=1, beta=0):
            tmp = image.astype(float) * alpha + beta
            tmp[tmp < 0] = 0
            tmp[tmp > 255] = 255
            image[:] = tmp
        image = image.copy()
        if random.randrange(2):
            #brightness distortion
            if random.randrange(2):
                _convert(image, beta=random.uniform(-32, 32))
            #contrast distortion
            if random.randrange(2):
                _convert(image, alpha=random.uniform(0.5, 1.5))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            #saturation distortion
            if random.randrange(2):
                _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))
            #hue distortion
            if random.randrange(2):
                tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
                tmp %= 180
                image[:, :, 0] = tmp
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image

class RandomErasing(object):
    def __init__(self, EPSILON=0.5, sl=0.02, sh=0.4, r1=0.3,
                 mean=[0., 0., 0.]):
        self.EPSILON = EPSILON
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.EPSILON:
            return img

        for attempt in range(100):
            area = img.shape[0] * img.shape[1]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            
            #
            if w < img.shape[0] and h < img.shape[1]:
                x1 = random.randint(0, img.shape[1] - h)
                y1 = random.randint(0, img.shape[0] - w)
                if img.shape[2] == 3:
                    img[ x1:x1 + h, y1:y1 + w, 0] = self.mean[0]
                    img[ x1:x1 + h, y1:y1 + w, 1] = self.mean[1]
                    img[ x1:x1 + h, y1:y1 + w, 2] = self.mean[2]
                else:
                    img[x1:x1 + h, y1:y1 + w,0] = self.mean[0]
                return img
            return img

class RandCropImage(object):
    def __init__(self, size, scale=None, ratio=None, interpolation=-1):

        self.interpolation = interpolation if interpolation >= 0 else None
        if type(size) is int:
            self.size = (size, size)  # (h, w)
        else:
            self.size = size

        self.scale = [0.08, 1.0] if scale is None else scale
        self.ratio = [3. / 4., 4. / 3.] if ratio is None else ratio

    def __call__(self, img):
        size = self.size
        scale = self.scale
        ratio = self.ratio

        aspect_ratio = math.sqrt(random.uniform(*ratio))
        w = 1. * aspect_ratio
        h = 1. / aspect_ratio

        img_h, img_w = img.shape[:2]

        bound = min((float(img_w) / img_h) / (w**2),
                    (float(img_h) / img_w) / (h**2))
        scale_max = min(scale[1], bound)
        scale_min = min(scale[0], bound)

        target_area = img_w * img_h * random.uniform(scale_min, scale_max)
        target_size = math.sqrt(target_area)
        w = int(target_size * w)
        h = int(target_size * h)

        i = random.randint(0, img_w - w)
        j = random.randint(0, img_h - h)

        img = img[j:j + h, i:i + w, :]
        if self.interpolation is None:
            return cv2.resize(img, size)
        else:
            return cv2.resize(img, size, interpolation=self.interpolation)

    
if __name__=='__main__':
    train_dataset = ReadDataset(cfg.shanghai_dir,image_sets=[('part_B_final', 'train_data')])