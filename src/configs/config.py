# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/12/20 17:09
#project: face anti spoofing
#company: 
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  face anti spoofing
####################################################
#from easydict import EasyDict 

#cfg = EasyDict()
class cfgs(object):
    def __init__(self):
        self.InputSize_w = None
        self.InputSize_h = None
        self.MAX_STEPS  = None
        self.Imgdir = None
        self.EPOCHES = None
        self.LR_STEPS = None
        self.LR_Values = None
        self.train_file = None
        self.test_file = None
        self.CLS_NUM = None
        self.FaceProperty = None
        self.train_mod = None
        self.MODEL_ASPP_OUTDIM = 512
        self.MODEL_SHORTCUT_DIM = 512
        self.csv_file = None
        
cfg = cfgs()
cfg.InputSize_w = 448 #224
cfg.InputSize_h = 448 #224
# shanghai dataset dir
cfg.Imgdir = '/home/lixiaoyu80/Datas/phone_detect' #'/mnt/data/LXY.data/img_celeba/img_detected' #
# training set
cfg.EPOCHES = 300
cfg.LR_Values = [0.00025,0.0005,0.0001,0.00005]
cfg.LR_STEPS = [100000,200000,300000,400000]
cfg.MAX_STEPS = 5000000
cfg.train_file = '/home/lixiaoyu80/Develop/mobilephone/data/train_crop_broken.txt' #'../data/train_all_c3.txt' train_broken.txt ssl_front2broken
cfg.test_file = '/home/lixiaoyu80/Develop/mobilephone/data/val_crop_broken.txt' #'../data/val_all_c3.txt' val_broken.txt test_broken_p24
cfg.csv_file = '/home/lixiaoyu80/Develop/mobilephone/data/broken_ssl_ecaresnet101dv2_result.csv'
# -------------------------------------------- test model
'''
cfg.threshold = [0.5,0.6,0.8]
cfg.ShowImg = 0
cfg.debug = 0
cfg.display_model = 0
cfg.batch_use = 0
cfg.time = 0
cfg.x_y = 1
cfg.box_widen = 1
'''
#-------------------------------------face attribute
cfg.CLS_NUM = 2 #21 #inlcude background:0, mobile:1  tv:2 remote-control:3
cfg.FaceProperty = ["unbroken","broken"] #["unbroken","broken"] #['background','back','front'] #['background','back','front','computer','Ipad','TV']['No_Beard','Mustache','Goatee','5_o_Clock_Shadow','Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair','Bangs','Bald', \
        # 'Male','Wearing_Hat','Wearing_Earrings','Wearing_Necklace','Wearing_Necktie',\
        # 'Eyeglasses','Young','Smiling','Arched_Eyebrows','Bushy_Eyebrows','Blurry']
        #['normal','unsell']

# shanghai dataset dir
#cfg.img_dir = '/home/lixiaoyu80/Datas'  #'/mnt/data/LXY.data/'
#cfg.model_dir = '/home/lixiaoyu80/models/mobilephone' #'/data/lxy/models/head'
# training set
#cfg.train_mod = 0 # data generate one by one
#cfg.ClsNum = 2
cfg.MODEL_ASPP_OUTDIM = 512
cfg.MODEL_SHORTCUT_DIM = 512