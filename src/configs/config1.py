###############################################
#created by :  lxy
#Time:  2021/1/14 17:09
#project: mobilephone
#company: jd
#rversion: v1
#tool:   python 3.6
#modified:
#description  
####################################################
from easydict import EasyDict 

cfg = EasyDict()

cfg.InputSize_w = 448 #224
cfg.InputSize_h = 448 #224
# shanghai dataset dir
# training set
cfg.EPOCHES = 20
cfg.LR_STEPS = [100000,200000,300000,400000]
cfg.LR_Values = [0.001,0.0005,0.00001]
cfg.MAX_STEPS = 500000
cfg.train_file = '..\data\\train_crop_broken.txt'
cfg.test_file = '..\data\\val_crop_broken.txt'
# -------------------------------------------- test model
cfg.threshold = [0.5,0.6,0.8]
cfg.ShowImg = 0
cfg.debug = 0
cfg.display_model = 0
cfg.batch_use = 0
cfg.time = 0
cfg.x_y = 1
cfg.box_widen = 1
#-------------------------------------face attribute
cfg.CLS_NUM = 2 #21 #inlcude background:0, mobile:1  tv:2 remote-control:3
cfg.FaceProperty = ['unbroken','broken'] # ['No_Beard','Mustache','Goatee','5_o_Clock_Shadow','Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair','Bangs','Bald', \
        # 'Male','Wearing_Hat','Wearing_Earrings','Wearing_Necklace','Wearing_Necktie',\
        # 'Eyeglasses','Young','Smiling','Arched_Eyebrows','Bushy_Eyebrows','Blurry']
        #['normal','unsell']
# for keras
cfg.Imgdir = 'D:\Datas\mobilephone\phone_detect\phone_crops' #'/home/lixiaoyu80/Datas/p2_1_rp448'  #'/mnt/data/LXY.data/'
cfg.model_dir = 'D:\Datas\models' #'/home/lixiaoyu80/models/mobilephone' #'/data/lxy/models/head'
# training set
cfg.train_mod = 0 # data generate one by one
cfg.ClsNum = 2
# aspp
cfg.MODEL_ASPP_OUTDIM = 256
cfg.MODEL_SHORTCUT_DIM = 48