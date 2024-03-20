#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import sys
import os
import torch
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import cv2
import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import cm as CM

sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfg
sys.path.append(os.path.join(os.path.dirname(__file__),'../networks'))
# from csr import CSRNet
from model import CSRNet

def parms():
    parser = argparse.ArgumentParser(description='CSRnet demo')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory for detect result')
    parser.add_argument('--modelpath', type=str,
                        default='weights/s3fd.pth', help='trained model')
    parser.add_argument('--threshold', default=0.65, type=float,
                        help='Final confidence threshold')
    parser.add_argument('--ctx', default=True, type=bool,
                        help='gpu run')
    parser.add_argument('--img_dir', type=str, default='tmp/',
                        help='Directory for images')
    parser.add_argument('--file_in', type=str, default='tmp.txt',
                        help='image namesf')
    parser.add_argument('--file_out', type=str, default='tmp.txt',
                        help='image namesf')
    return parser.parse_args()


class HeadCount(object):
    def __init__(self,args):
        if args.ctx and torch.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False
        if self.use_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
        self.loadmodel(args.modelpath)
        self.threshold = args.threshold
        self.img_dir = args.img_dir
        self.real_num = 0
        self.file_out = args.file_out
        self.kernel_size = 55
        self.save_dir = args.save_dir
        if self.save_dir is not None:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

    def loadmodel(self,modelpath):
        if self.use_cuda:
            device = 'cuda'
        else:
            device = 'cpu'
        self.net = CSRNet(self.use_cuda)
        self.net.load_state_dict(torch.load(modelpath,map_location=device))
        self.net.eval()
        if self.use_cuda:
            self.net.cuda()
            cudnn.benckmark = True

    def display_hotmap(self,img,hotmaps):
        '''
        hotmaps: a list of hot map ,every shape is [1,h,w]
        '''  
        pred_map = hotmaps[0]
        # pred_map = pred_map/np.max(pred_map+1e-20) 
        pred_num = str(int(np.sum(hotmaps[0])))
        txt = "real_num:%s - pred_num:%s" % (str(self.real_num),pred_num)
        fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
        ax1 = axes[0]
        ax1.imshow(img[:,:,::-1])
        ax1.set_title(txt)
        ax2 = axes[1]
        pred_map = self.get_colors(pred_map)
        ax2.imshow(pred_map[:,:,::-1]) #cmap='jet'
        plt.savefig('mm1.png')
        # plt.savefig(self.savepath+'.png')
        # plt.title(txt)
        # plt.show()
        # pred_frame = plt.gca()
        # plt.imshow(pred_map, 'jet')
        # pred_frame.axes.get_yaxis().set_visible(False)
        # pred_frame.axes.get_xaxis().set_visible(False)
        # pred_frame.spines['top'].set_visible(False) 
        # pred_frame.spines['bottom'].set_visible(False) 
        # pred_frame.spines['left'].set_visible(False) 
        # pred_frame.spines['right'].set_visible(False) 
        # plt.savefig(exp_name+'/'+filename_no_ext+'_pred_'+str(float(pred))+'.png',\
        #     bbox_inches='tight',pad_inches=0,dpi=150)
        plt.close()

    def get_colors(self,hotmap):
        indexs = np.where(hotmap>1e-5)
        h,w = hotmap.shape[:2]
        colormap = np.zeros((h,w,3),dtype=np.uint8)
        keep_map = hotmap[indexs]
        max_v = np.max(keep_map)
        min_v = np.min(keep_map)
        # print('min max,',min_v,max_v)
        # total_num = len(indexs[0])
        # print(total_num)
        # for i in range(total_num):
        #     iy = indexs[0][i]
        #     ix = indexs[0][i]
        #     tmp = hotmap[iy,ix]
        #     colormap[iy,ix,2] = np.uint8((tmp-min_v+1e-6)/(max_v -min_v) * 255)
        hotmap = hotmap*10000
        hotmap = np.maximum(hotmap,0)
        hotmap = np.minimum(hotmap,255)
        colormap[:,:,2]=hotmap.astype(np.uint8)
        return colormap
    def apply_density(self,img,hotmap):
        # create a blank img
        h,w,_ = img.shape
        ih,iw = hotmap.shape[:2]
        # img = cv2.resize(img,(iw,ih))
        overlay = img.copy()
        pred_num = str(int(np.sum(hotmap)))
        point = (int(w-300),20)
        keep_indx = np.where(hotmap>0.0001)
        alpha = 0.5
        cv2.rectangle(overlay, (0, 0), (img.shape[1], img.shape[0]), (255, 0, 0), -1) 
        for i in range(len(keep_indx[0])):
            # iy = np.clip(keep_indx[0][i]/float(ih) * h,0,h-1)
            # ix = np.clip(keep_indx[1][i]/float(iw) * w,0,w-1)
            ix = keep_indx[1][i]
            iy = keep_indx[0][i]
            cv2.circle(overlay,(int(ix),int(iy)),3,(0,0,255))
        # image = cv2.addWeighted(overlay, alpha, image, 1-alpha, 0) 
        image = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0) 
        txt = "real_num:%s - pred_num:%s" % (str(self.real_num),pred_num)
        cv2.putText(image,txt,point,cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
        return image
        
    def propress(self,img):
        # rgb_mean = np.array([123.,117.,104.])[np.newaxis, np.newaxis,:].astype('float32')
        # rgb_mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis,:].astype('float32')
        # rgb_std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis,:].astype('float32')
        rgb_mean = np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis,:].astype('float32')
        rgb_std = np.array([0.225, 0.225, 0.225])[np.newaxis, np.newaxis,:].astype('float32')
        #img = cv2.resize(img,(cfg.resize_width,cfg.resize_height))
        # img = self.rescaleimg(img,2048)
        h,w = img.shape[:2]
        gth = int(np.ceil(h/8.0)*8)
        gtw = int(np.ceil(w/8.0)*8)
        img = cv2.resize(img,(gtw,gth))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = img.astype('float32')
        img /= 255.0
        img -= rgb_mean
        img /= rgb_std
        img = np.transpose(img,(2,0,1))
        return img

    def rescaleimg(self,img,maxsize):
        h,w = img.shape[:2]
        tmax = max(h,w)
        if tmax > maxsize:
            scale = maxsize / float(tmax)
            nh = int(h*scale)
            nw = int(w*scale)
            img = cv2.resize(img,(nw,nh))
        return img

    def get_boxarea(self,img,frame):
        '''
        img: gray img
        '''
        img = img[0]
        dencity_map = img.copy()
        imgh,imgw = img.shape[:2]
        frameh,framew = frame.shape[:2]
        # print('min',np.min(img))
        # print('max',np.max(img))
        # img = np.where(img >0.0002,255,0)
        _,img = cv2.threshold(img,0.0002,255,cv2.THRESH_BINARY)
        img = np.array(img,dtype=np.uint8)
        # cv2.imshow('thresh',img)
        kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size,1 ))
        kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, self.kernel_size))
        img = cv2.dilate(img, kernelX, iterations=2)
        img = cv2.erode(img, kernelX,  iterations=4)
        img = cv2.dilate(img, kernelX,  iterations=2)
        img = cv2.erode(img, kernelY,  iterations=1)
        img = cv2.dilate(img, kernelY,  iterations=2)
        img = cv2.medianBlur(img, 3)
        # img = cv2.medianBlur(img, 15)
        # cv2.imshow('dilate&erode', img)
        #输入的三个参数分别为：输入图像、层次类型、轮廓逼近方法
        #返回的三个返回值分别为：修改后的图像、图轮廓、层次
        # image, contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        hull = []
        for i ,c in enumerate(contours):
            # 边界框
            x, y, w, h = cv2.boundingRect(c)
            hull.append(cv2.convexHull(c, False))
            if min(w,h) > 100:
                # x2 = int((x+w)/float(imgw) * framew)
                # y2 = int((y+h)/float(imgh) * frameh)
                # x1 = int(x/imgw *framew)
                # y1 = int(y/imgh *frameh)
                x1,x2,y1,y2 = int(x),int(x+w),int(y),int(y+h)
                # tmp = int(np.sum(dencity_map[x1:x2+1,y1:y2+1]))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.putText(frame,str(tmp),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
                boxes.append([x1,y1,x2,y2])
                cv2.polylines(frame, [hull[i]], True, (0, 255, 0), 2)
            # cv2.drawContours(frame, hull, i, (0, 255, 0), 1, 8)
            # cv2.polylines(frame, [hull[i]], True, (0, 255, 0), 2)
        return frame,boxes

    def inference_img(self,imgorg):
        # t1 = time.time()
        # h,w = imgorg.shape[:2]
        img = self.propress(imgorg.copy())
        bt_img = torch.from_numpy(img).unsqueeze(0)
        if self.use_cuda:
            bt_img = bt_img.cuda()
        output = self.net(bt_img)
        cnt_num = torch.sum(output[0])
        cnt_num = cnt_num.data.cpu().numpy()
        output = output.data.cpu().numpy()
        # t2 = time.time()
        # print('consuming:',t2-t1)
        if cnt_num >1:
            img_out = self.apply_density(imgorg,output[0])
            self.display_hotmap(imgorg.copy(),output)
            img_density,boxes = self.get_boxarea(output,img_out)
            return img_density,cnt_num
        else:
            return imgorg,0

    def rescalmask(self,imglist,imgw,imgh):
        imgout = []
        for i in range(imglist.shape[0]):
            # print(imglist[i].shape)
            tmp = cv2.resize(imglist[i],(imgw,imgh),cv2.INTER_NEAREST)
            imgout.append(tmp)
        return np.array(imgout)

    def headcnts(self,imgpath):
        if os.path.isdir(imgpath):
            cnts = os.listdir(imgpath)
            total_num = len(cnts)
            for i in tqdm(range(total_num)):
                tmp = cnts[i].strip()
                tmppath = os.path.join(imgpath,tmp)
                imgname = tmp
                # self.savepath = os.path.join(self.save_dir,imgname)
                img = cv2.imread(tmppath)
                if img is None:
                    continue
                img = self.rescaleimg(img,2048)
                h,w = img.shape[:2]
                gth = int(np.ceil(h/8.0)*8)
                gtw = int(np.ceil(w/8.0)*8)
                img = cv2.resize(img,(gtw,gth))
                frame,cnt_head = self.inference_img(img)
                # print('heads >> ',cnt_head)
                cv2.imshow('result',frame)
                # cv2.imwrite(self.savepath,frame)
                cv2.waitKey(0) 
        elif os.path.isfile(imgpath) and imgpath.endswith('txt'):
            # if not os.path.exists(self.save_dir) and self.save_dir is not None:
                # os.makedirs(self.save_dir)
            f_r = open(imgpath,'r')
            f_w = open(self.file_out,'w')
            file_cnts = f_r.readlines()
            for j in tqdm(range(len(file_cnts))):
                tmp_file = file_cnts[j].strip()
                imgname = tmp_file
                # tmp_file_s = tmp_file.split('\t')
                # if len(tmp_file_s)>0:
                    # tmp_file = tmp_file_s[0]
                    # self.real_num = int(tmp_file_s[1])
                if not tmp_file.endswith('jpg'):
                    tmp_file = tmp_file +'.jpg'
                tmp_path = os.path.join(self.img_dir,tmp_file) 
                # tmp_path = tmp_file
                # save_name = tmp_file
                if not os.path.exists(tmp_path):
                    print(tmp_path)
                    continue
                img = cv2.imread(tmp_path)
                # img = self.rescaleimg(img,2048) 
                if img is None:
                    print('None',tmp_path)
                    continue
                # self.savepath = os.path.join(self.save_dir,save_name)
                frame,cnt_head = self.inference_img(img)
                # cv2.imshow('result',frame)
                # cv2.imwrite(self.savepath,frame)
                # cv2.waitKey(10) 
                f_w.write("{} {:.4f}\n".format(imgname,cnt_head))
            f_r.close()
            f_w.close()
        elif os.path.isfile(imgpath) and imgpath.endswith(('.mp4','.avi')) :
            cap = cv2.VideoCapture(imgpath)
            frame_width =  cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame_height =  cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            # out = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 25,(frame_width, frame_height))
            if not cap.isOpened():
                print("failed open camera")
                return 0
            else: 
                while cap.isOpened():
                    _,img = cap.read()
                    frame,cnt_head = self.inference_img(img)
                    # out.write(frame)
                    # cv2.imshow('result',frame)
                    q=cv2.waitKey(10) & 0xFF
                    # cv2.imwrite('test_video1.jpg',frame)
                    if q == 27 or q ==ord('q'):
                        break
            cap.release()
            cv2.destroyAllWindows()
        elif os.path.isfile(imgpath):
            img = cv2.imread(imgpath)
            h,w = img.shape[:2]
            gth = int(np.ceil(h/8.0)*8)
            gtw = int(np.ceil(w/8.0)*8)
            img = cv2.resize(img,(gtw,gth))
            imgname = imgpath.split('/')[-1].strip()
            if img is not None:
                # grab next frame
                # update FPS counter
                frame,cnt_head = self.inference_img(img)
                # hotmaps = self.get_hotmaps(odm_maps)
                # self.display_hotmap(hotmaps)
                # keybindings for display
                print(cnt_head)
                # frame = np.where(frame[0]>0.0005,255,0)
                # frame = np.array(frame,dtype=np.uint8)
                cv2.imshow('result',frame)
                cv2.imwrite(imgname,frame)
                key = cv2.waitKey(0) 
        else:
            print('please input the right img-path')

if __name__ == '__main__':
    args = parms()
    detector = HeadCount(args)
    imgpath = args.file_in
    detector.headcnts(imgpath)
