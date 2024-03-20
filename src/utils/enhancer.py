import numpy as np 
from matplotlib import pyplot as plt 
import os
import sys
import cv2
import tqdm
import shutil
import csv
import random


def cropfromdir(imgdir,savedir,cropsize=448):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    fcnts = os.listdir(imgdir)
    total_num = len(fcnts)
    cnt = 0
    for i in tqdm.tqdm(range(total_num)):
        tmp = fcnts[i].strip()
        imgpath = os.path.join(imgdir,tmp)
        if not os.path.exists(imgpath):
            print(imgpath)
            continue
        img = cv2.imread(imgpath)
        imgh,imgw = img.shape[:2]
        if imgh > cropsize and imgw > cropsize:
            choicew = imgw - cropsize
            choiceh = imgh - cropsize
            for i in range(2):
                savename = 'bg'+'_'+str(cnt)+'_'+tmp
                savepath = os.path.join(savedir,savename)
                pointx = random.randint(0,choicew)
                pointy = random.randint(0,choiceh)
                x2 = pointx + cropsize
                y2 = pointy + cropsize
                tmpimg = img[pointy:y2,pointx:x2,:]
                cv2.imwrite(savepath,tmpimg)
                cnt+=1
        else:
            savename = 'bg'+'_'+str(cnt)+'_'+tmp
            savepath = os.path.join(savedir,savename)
            shutil.copyfile(imgpath,savepath)
            cnt+=1

if __name__ == '__main__':
    imgdir ='D:\Datas\mobilephone\phone_detect\phone_crops\p24_crops\\unbroken_imgs'
    savedir = 'D:\Datas\mobilephone\phone_detect\phone_crops\\broken_crops\p24_unbroken'
    cropfromdir(imgdir,savedir)