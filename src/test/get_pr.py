# -*- coding: utf-8 -*-
###############################################
#created by :  lixiaoyu80
#Time:  2021/01/12 19:09
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
import cv2
import argparse
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from matplotlib.pyplot import MultipleLocator
import os 
import sys
import csv
import json
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfg

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def args():
    parser = argparse.ArgumentParser(description="mtcnn caffe")
    parser.add_argument('--file-in',type=str,dest='file_in',default='None',\
                        help="the file input path")
    parser.add_argument('--out-file',type=str,dest='out_file',default='None',\
                        help="the file output path")
    parser.add_argument('--base-dir',type=str,dest='base_dir',default="./base_dir",\
                        help="images saved dir")
    parser.add_argument('--gpu', default=None, type=str,help='which gpu to run')
    parser.add_argument('--record-file',type=str,dest='record_file',default=None,\
                        help="output saved file")
    parser.add_argument('--cmd-type', default="dbtest",dest='cmd_type', type=str,\
                        help="which code to run: videotest,imgtest ")
    parser.add_argument('--name', default='resnet50', type=str,help='Final confidence threshold')
    return parser.parse_args()


def OneHotlabel(label_idx,class_num):
    '''
    labels: cls_num
    return onehotlabel
    '''
    onehotlabel = np.zeros([class_num])
    onehotlabel[int(label_idx)] = 1
    return onehotlabel

class SaveCSV(object):
    def __init__(self,savepath,keyword_list):
        '''
        save csv file
        '''
        # 第一次打开文件时，第一行写入表头
            # with open(savepath, "w", newline='', encoding='utf-8') as csvfile:  # newline='' 去除空白行
        self.csvfile = open(savepath, "w", newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.csvfile, fieldnames=keyword_list)  # 写字典的方法
        self.writer.writeheader()  # 写表头的方法
        print("creat csv saver")

    def save(self,itemnew):
        """
        保存csv方法
        :param keyword_list: 保存文件的字段或者说是表头
        :param path: 保存文件路径和名字
        :param item: 要保存的字典对象
        :return:
        """
        try:
            # 接下来追加写入内容
            # with open(savepath, "a", newline='', encoding='utf-8') as csvfile:  # newline='' 一定要写，否则写入数据有空白行
            # writer = csv.DictWriter(csvfile, fieldnames=keyword_list)
            self.writer.writerow(itemnew)  # 按行写入数据
                # print("^_^ write success")

        except Exception as e:
            print("write error==>", e)
            # 记录错误数据
            with open("error.txt", "w") as f:
                f.write(json.dumps(itemnew) + ",\n")
            pass
    def close(self):
        self.csvfile.close()

def evalue(args):
    '''
    file_in: csv data
    out_file: csv data
    calculate the tpr and fpr for all classes
    real positive: tp+fn
    real negnitive: fp+tn
    R = tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    P = tp/(tp+fp)
    '''
    file_in = args.file_in
    result_out = args.out_file
    if file_in is None:
        print("input file is None",file_in)
        return None
    key_list = ['threshold']
    for tmp in cfg.FaceProperty:
        key_list.append(tmp+"_tpr")
        key_list.append(tmp+'_precision')
        key_list.append(tmp+'_fpr')
    record_w = SaveCSV(result_out,key_list)
    for tmp_threshold in np.arange(0.2,1.0,0.01):
        file_rd = open(file_in,'r')
        reader = csv.DictReader(file_rd)
        statistics_dic = defaultdict(lambda : 0)
        tmp_dict = dict()
        tmp_dict['threshold'] = tmp_threshold
        for i,tmp_item in enumerate(reader):
            imgpath = tmp_item['filename']
            name_spl = imgpath.split('\\')
            # if 'back' == name_spl[0]:
            #    real_label = 1
            # elif 'pos_samples' == name_spl[0]:
            #    real_label = 2
            # else:
            #    real_label = 0
            if 'unbroken' in  imgpath: #name_spl[0]:
                real_label = 0
            else:
                real_label = 1
            cur_data = []
            for cur_key in cfg.FaceProperty:
                cur_data.append(float(tmp_item[cur_key]))
            pred_id = np.argmax(np.array(cur_data))
            if pred_id ==1: # here is different from the original**************
                tmp_score = cur_data[pred_id]
                if tmp_score<= tmp_threshold:
                    cur_data[pred_id] = 0.0
                    pred_id = np.argmax(cur_data)
            real_label = OneHotlabel(real_label,cfg.CLS_NUM)
            pred_ids = OneHotlabel(pred_id,cfg.CLS_NUM)
            score = cur_data[pred_id]
            # tmp_item = dict()
            # key_id = 0
            # tmp_item[key_list[key_id]]=img_name
            for idx in range(cfg.CLS_NUM):
                pred_cls_id = pred_ids[idx]
                real_cls_id = int(real_label[idx])
                pred_name = cfg.FaceProperty[idx]
                real_name = cfg.FaceProperty[idx]
                if real_cls_id:
                    statistics_dic[real_name+'_tpfn'] +=1
                else:
                    statistics_dic[real_name+'_fptn'] +=1          
                if int(pred_cls_id) == int(real_cls_id)==1:
                    statistics_dic[pred_name+'_tp'] +=1
                elif int(pred_cls_id)==int(real_cls_id)==0:
                    statistics_dic[pred_name+'_tn'] +=1
                # display(img_data,idx)
            # record_w.save(tmp_item)
        # file_wr.write("cls_name,tpr,fpr,precision,tp_fn,fp_tn,tp,fp\n")
        
        for key_name in cfg.FaceProperty:
            tp_fn = statistics_dic[key_name+'_tpfn']
            tp = statistics_dic[key_name+'_tp']
            tn = statistics_dic[key_name+'_tn']
            fp_tn = statistics_dic[key_name+'_fptn']
            fp = fp_tn - tn
            tpr = float(tp) / tp_fn if tp_fn else 0.0
            fpr = float(fp) / fp_tn if fp_tn else 0.0
            precision = float(tp) / (tp+fp) if tp+fp else 0.0
            # statistics_dic[key_name+'_tpr'] = tpr
            # statistics_dic[key_name+'_fpr'] = fpr
            # statistics_dic[key_name+'_P'] = precision
            tmp_dict[key_name+'_tpr'] = tpr
            tmp_dict[key_name+'_fpr'] = fpr
            tmp_dict[key_name+'_precision'] = precision
            # file_wr.write('>>> {} result is: tp_fn-{} | fp_tn-{} | tp-{} | fp-{}\n'.format(key_name,\
                        #    tp_fn,fp_tn,tp,fp))
            # file_wr.write('\t tpr:{:.4f} | fpr:{:.4f} | Precision:{:.4f}\n'.format(tpr,fpr,precision))
            # file_wr.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(key_name,tpr,fpr,precision,tp_fn,fp_tn,tp,fp))
        record_w.save(tmp_dict)
        file_rd.close()

    # file_rd.close()
    # record_w.close()

def plot_lines(txt_path,name):
    '''
    filein: csv file. threshold,key_name,tpr,precision,fpr
    '''
    ax_data,total_data = readdata(txt_path)
    fig = plt.figure(num=0,figsize=(20,10))
    plt.plot(ax_data,total_data[-1],label='total' )
    plt.plot(ax_data,total_data[0],label='loc')
    plt.plot(ax_data,total_data[1],label='conf')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('%s_training' % name)
    plt.grid(True)
    leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.2)
    #leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    plt.savefig("./logs/%s.png" % name ,format='png')
    plt.show()

def plot_data(filein,name):
    '''
    datadict: keyname_tpr_data,keyname_frp_data,keyname_precision_data
    ax_data: threshold data list
    '''
    row_num = 1 #4
    col_num = 2 #6
    #keys_list = ['front','back','background']
    keys_list = ['unbroken','broken']
    datadict,ax_data = readdata(filein,keys_list)
    fig, axes = plt.subplots(nrows=row_num, ncols=col_num,figsize=(20,20))
    #plt.rcParams.update({"font.size":4})
    axes.flatten()
    #plt.tick_params(labelsize=5)
    #plt.xticks(fontproperties = 'Times New Roman', size = 2)
    #plt.yticks(fontproperties = 'Times New Roman', size = 2)
    y_major_locator=MultipleLocator(0.01)
    x_major_locator=MultipleLocator(0.1)
    #ax_data = np.array(ax_data)
    #idx = np.where(ax_data>=0.5)[0]
    #ax_data = ax_data[idx]
    #ax_data = ax_data[:-4]
    for i in range(row_num):
        for j in range(col_num):
            keyname = keys_list[j]
            #keyname = 'broken'
            print(keyname)
            tpr_data = datadict[keyname+'_tpr']
            fpr_data = datadict[keyname+'_fpr']
            p_data = datadict[keyname+'_precision']
            #tpr_data = tpr_data[idx]
            #fpr_data = fpr_data[idx]
            #p_data = p_data[idx]
            axes[j].plot(ax_data,tpr_data,label='recall')
            axes[j].plot(ax_data,fpr_data,label='fpr')
            axes[j].plot(ax_data,p_data,label='precision')
            axes[j].set_title(keyname,fontdict={'family' : 'Times New Roman', 'size'   : 6})
            axes[j].set_xlabel('threshold')
            axes[j].yaxis.set_major_locator(y_major_locator)
            axes[j].xaxis.set_major_locator(x_major_locator)
            #axes.axvline(x=0.85,ls="-",c="red")
            #axes.yaxis.set_major_locator(y_major_locator)
            axes[j].grid(True)
            axes[j].legend()
            # if (i*col_num +j+1)== total_num:
            #     break
            #auc = get_auc(fpr_data[:-3],tpr_data[:-3])
            #print("*******auc:",keyname,auc)
            #auc_txt = "auc:%.3f" % auc
            #plt.text(fpr_data[50],tpr_data[50],auc_txt,fontsize=15)
    plt.savefig('../data/%s.png' % name,format='png')
    plt.show()
    
def plot_roc(filein,name):
    '''
    datadict: keyname_tpr_data,keyname_frp_data,keyname_precision_data
    ax_data: threshold data list
    '''
    row_num = 1 #4
    col_num = 1 #6
    #keys_list = ['front','back','background']
    keys_list = ['broken','unbroken']
    datadict,ax_data = readdata(filein,keys_list)
    fig, axes = plt.subplots(nrows=row_num, ncols=col_num,figsize=(20,20))
    #plt.rcParams.update({"font.size":4})
    #axes.flatten()
    #plt.tick_params(labelsize=5)
    #plt.xticks(fontproperties = 'Times New Roman', size = 2)
    #plt.yticks(fontproperties = 'Times New Roman', size = 2)
    y_major_locator=MultipleLocator(0.01)
    x_major_locator=MultipleLocator(0.05)
    #ax_data = np.array(ax_data)
    #idx = np.where(ax_data>=0.5)[0]
    #ax_data = ax_data[idx]
    #ax_data = ax_data[:-4]
    for i in range(row_num):
        for j in range(col_num):
            keyname = keys_list[j]
            print(keyname)
            tpr_data = datadict[keyname+'_tpr']
            fpr_data = datadict[keyname+'_fpr']
            p_data = datadict[keyname+'_precision']
            #tpr_data = tpr_data[idx]
            #fpr_data = fpr_data[idx]
            #p_data = p_data[idx]
            axes.plot(fpr_data,tpr_data,label='ROC') 
            #axes.plot(ax_data,tpr_data,label='Recall')
            #axes.plot(ax_data,fpr_data,label='FPR')
            #axes.plot(ax_data,p_data,label='Precision')
            #axes.set_title(keyname,fontdict={'family' : 'Times New Roman', 'size'   : 6})
            #axes.set_xlabel('threshold')
            axes.set_xlabel('FPR')
            axes.set_ylabel('TPR')
            #axes.yaxis.set_major_locator(y_major_locator)
            #axes.xaxis.set_major_locator(x_major_locator)
            #axes.axvline(x=0.85,ls="-",c="red")
            #axes.yaxis.set_major_locator(y_major_locator)
            axes.grid(True)
            axes.legend()
            # if (i*col_num +j+1)== total_num:
            #     break
            
            auc = get_auc(fpr_data[:-3],tpr_data[:-3])
            print("*******auc:",keyname,auc)
            auc_txt = "auc:%.3f" % auc
            plt.text(fpr_data[50],tpr_data[50],auc_txt,fontsize=15)
            
    plt.savefig('../data/%s.png' % name,format='png')
    plt.show()

def plot_single(filein,name):
    '''
    datadict: keyname_tpr_data,keyname_frp_data,keyname_precision_data
    ax_data: threshold data list
    '''
    keys_list = ['broken','unbroken']
    datadict,ax_data = readdata(filein,keys_list)
    fig, axes = plt.subplots(nrows=row_num, ncols=col_num,figsize=(20,20))
    #plt.rcParams.update({"font.size":4})
    #axes.flatten()
    #plt.tick_params(labelsize=5)
    #plt.xticks(fontproperties = 'Times New Roman', size = 2)
    #plt.yticks(fontproperties = 'Times New Roman', size = 2)
    y_major_locator=MultipleLocator(0.01)
    x_major_locator=MultipleLocator(0.05)
    #ax_data = np.array(ax_data)
    #idx = np.where(ax_data>=0.5)[0]
    #ax_data = ax_data[idx]
    #ax_data = ax_data[:-4]
    keyname = keys_list[0]
    print(keyname)
    tpr_data = datadict[keyname+'_tpr']
    fpr_data = datadict[keyname+'_fpr']
    p_data = datadict[keyname+'_precision']
    #tpr_data = tpr_data[idx]
    #fpr_data = fpr_data[idx]
    #p_data = p_data[idx] 
    axes.plot(ax_data,tpr_data,label='Recall')
    axes.plot(ax_data,fpr_data,label='FPR')
    axes.plot(ax_data,p_data,label='Precision')
    axes.set_title(keyname,fontdict={'family' : 'Times New Roman', 'size'   : 6})
    axes.set_xlabel('threshold')
    axes.yaxis.set_major_locator(y_major_locator)
    axes.xaxis.set_major_locator(x_major_locator)
    #axes.axvline(x=0.85,ls="-",c="red")
    #axes.yaxis.set_major_locator(y_major_locator)
    axes.grid(True)
    axes.legend()      
    plt.savefig('../data/%s.png' % name,format='png')
    plt.show()

def get_auc(mrec,mpre):
    # correct AP calculation
    # first append sentinel values at the end
    #mrec = np.concatenate(([0.], rec, [1.]))
    #mpre = np.concatenate(([0.], prec, [0.]))
    mrec.append(0.0)
    mpre.append(0.0)
    mrec.insert(0,1.0)
    mpre.insert(0,1.0)
    mrec=np.array(mrec)
    mpre=np.array(mpre)
    idx = np.argsort(mrec)
    mrec = mrec[idx]
    mpre = mpre[idx]
    # compute the precision envelope
    total_num = len(mpre)
    for i in range(total_num - 1, 0, -1):
        mpre[i - 1] = np.minimum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def readdata(filein,key_list):
    '''
    filein: csv file. threshold,key_name,tpr,precision,fpr
    return: data_dict,data_list
    '''
    data_dict = defaultdict(list)
    ax_data = []
    #key_list = ['front','back','background']
    fr = open(filein,'r')
    reader = csv.DictReader(fr)
    for tmp_item in reader:
        ax_data.append(float(tmp_item['threshold']))
        cnt = 0
        for tmp in key_list:
            data_dict[tmp+'_tpr'].append(float(tmp_item[tmp+'_tpr']))
            data_dict[tmp+'_fpr'].append(float(tmp_item[tmp+'_fpr']))
            data_dict[tmp+'_precision'].append(float(tmp_item[tmp+'_precision']))
    fr.close()
    return data_dict,ax_data



if __name__ == '__main__':
    parms = args()
    cmd_type = parms.cmd_type
    if cmd_type == 'plotscore':
        plot_data(parms.file_in,parms.name)
        #plot_roc(parms.file_in,parms.name)
    elif cmd_type == 'plotroc':
        plot_roc(parms.file_in,parms.name)
    elif cmd_type in 'evalue':
        evalue(parms)
    elif cmd_type in 'single':
        plot_single(parms.file_in,parms.name)
    else:
        print('Please input right cmd')