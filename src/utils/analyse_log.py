# -*- coding: utf-8 -*-
###############################################
#created by :  lixiaoyu80
#Time:  2021/02/17 11:09
#project: Mobilephone recognition
#company: jd
#rversion: v1
#tool:   python 3.6
#modified:
#description  
#citation
####################################################
import numpy as np
from collections import defaultdict 
import time 
import argparse
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
import os 
import sys
import csv
import json
from tqdm import tqdm
import ast

def SaveJson(cnts,savepath):
    '''
    cnts: dict 
    '''
    jstr = json.dumps(cnts)
    fw = open(savepath,'w')
    fw.write(jstr)
    fw.close()

def ReadJson(filein):
    '''
    filein: json file or a string
    return: dict
    '''
    if os.path.isfile(filein):
        fr = open(filein,'r',encoding='utf-8')
        fcnts = fr.read()
        js_dict = json.loads(fcnts)
    else:
        js_dict = ast.literal_eval(filein.strip())
    return js_dict

def ExtractPathsfromlog(logpath,savepath):
    '''
    logpath: log files producted online
    savepath: json file save path
    '''
    fr = open(logpath,'r',encoding='utf-8')
    # fcnts = fr.readlines()
    out_dict = dict()
    out_dict["channelCode"] = "FGHT"
    out_dict["extInfo"] = "extInfo"
    out_dict[ "aiImageInput1"] = "xxxx"
    # out_dict[ "aiImageInput"] = list()
    imginfo_list = []
    cnt = 0
    for tmp in fr:
        tmp = tmp.strip()
        # print(tmp)
        if "结果校验" in tmp:
            tmp_spl = tmp.split(">>")
            if len(tmp_spl)>0:
                # print(tmp_spl[2])
                img_dict = dict()
                tmp_dict = ReadJson(tmp_spl[2])
                # print(tmp_dict['objectName'])
                if tmp_dict["isfront"] =='1' and float(tmp_dict['frontProb']) > 0.85 and  tmp_dict["isBroken"]=='0': #and float(tmp_dict['brokenProb']) >=0.96: #in ['0','1']: #len(tmp_dict['imeiValue'])>0:
                    img_dict["imageModule"] =  tmp_dict["imageModule"] #"model_1"
                    img_dict[ "imageType"] =  tmp_dict[ "imageType"] # "type_1"
                    img_dict["objectName"] = tmp_dict["objectName"] 
                    img_dict["bucketName"] = tmp_dict["bucketName"] # "ins-smp"
                    cnt+=1
                    imginfo_list.append(img_dict)
    out_dict["aiImageInput"] = imginfo_list
    SaveJson(out_dict,savepath)
    print(cnt)

def checknames(dir1,dir2):
    fcnts = os.listdir(dir1)
    f2 = os.listdir(dir2)
    cnt = 0
    for tmp in f2:
        if tmp.strip() in fcnts:
            cnt+=1
    print('dir1:',len(fcnts))
    print('dir2',len(f2))
    print('iou',cnt)

if __name__=='__main__':
    # ExtractPathsfromlog('D:\Datas\online-log-org\logs\ins-bdg-414.log','D:\Datas\online-logs\online_fg_ubk414.json')
    checknames('D:\Datas\\test_online\onlinev7\online_fg_ubk','D:\Datas\\test_online\onlinev7\error\\bg2fg')
