import numpy as np
from collections import defaultdict 
import time 
import cv2
import argparse
import os 
import sys
import csv
import json
from tqdm import tqdm


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

def Merge2modelResult(file1,file2,outfile):
    '''
    file1: model 1 output,f1.csv
    file2: model 2 output, f2.csv
    outfile: file1 or file2, exp: imagepath, reallabel,pred_id,unbroken_score,broken_score
    '''
    filewr = open(outfile,'w')
    filewr.write("imagepath,reallabel,pred_id,unbroken_score,broken_score\n")
    def getdata(file_in):
        file_rd = open(file_in,'r')
        reader = csv.DictReader(file_rd)
        dictout = defaultdict(list)
        for f_item in reader:
            #print(f_item['filename'])
            tmprecord = []
            cur_data = []
            imgpath = f_item['filename']
            for cur_key in ['unbroken','broken']:
                cur_data.append(float(f_item[cur_key]))
            pred_id = np.argmax(np.array(cur_data))
            tmprecord.append(pred_id)
            tmprecord.extend(cur_data)
            dictout[imgpath]=tmprecord
        file_rd.close()
        return dictout
    dict1 = getdata(file1)
    dict2 = getdata(file2)
    imagenames = dict1.keys()
    for tmp in imagenames:
        cnt1 = dict1[tmp]
        cnt2 = dict2[tmp]
        if 'unbroken' in tmp:
            reallabel = 0
        else:
            reallabel = 1
        tmplist = []
        tmplist.append(reallabel)
        if int(cnt1[0])==1 and int(cnt2[0])==1:
            if cnt1[2] > cnt2[2]:
                tmplist.extend(cnt1)
            else:
                tmplist.extend(cnt2)
        elif int(cnt1[0])==1:
            tmplist.extend(cnt1)
        elif int(cnt2[0])==1:
            tmplist.extend(cnt2)
        else:
            if cnt1[1]> cnt2[1]:
                tmplist.extend(cnt1)
            else:
                tmplist.extend(cnt2)
        tmpstr = list(map(str,tmplist))
        outstr = ','.join(tmpstr)
        filewr.write("{},{}\n".format(tmp,outstr))
    filewr.close()

def OneHotlabel(label_idx,class_num):
    '''
    labels: cls_num
    return onehotlabel
    '''
    onehotlabel = np.zeros([class_num])
    onehotlabel[int(label_idx)] = 1
    return onehotlabel

def analysisfile(filein,outfile,recordfile):
    key_list = ['filename']
    properties = ['unbroken','broken']
    class_num = 2
    for tmp in properties:
        key_list.append(tmp)
        key_list.append(tmp+'_fg')
    record_w = SaveCSV(outfile,key_list)
    filerd = open(filein,'r')
    filewr = open(recordfile,'w')
    file_cnts = filerd.readlines()
    total_num = len(file_cnts)
    statistics_dic = defaultdict(lambda : 0)
    for i in tqdm(range(total_num)):
        if i==0:
            continue
        item_cnt = file_cnts[i]
        item_spl = item_cnt.strip().split(',')
        # print(item_spl[1:])
        img_name = item_spl[0]
        real_label_cls = item_spl[1]
        real_label = OneHotlabel(real_label_cls,class_num)
        pred_ids_cls = item_spl[2]
        pred_ids = OneHotlabel(pred_ids_cls,class_num)
        scores = [float(item_spl[3]),float(item_spl[4])]
        tmp_item = dict()
        key_id = 0
        tmp_item[key_list[key_id]]=img_name
        for idx in range(class_num):
            pred_cls_id = int(pred_ids[idx])
            real_cls_id = int(real_label[idx])
            pred_name = properties[idx]
            #real_name = cfgs.FaceProperty[int(real_label)]
            real_name = properties[idx]
            # tmp_item.append(str(scores[idx]))
            # tmp_item.append(real_label[idx])
            key_id+=1
            tmp_item[key_list[key_id]] = "%.3f" % scores[idx]
            key_id+=1
            tmp_item[key_list[key_id]] = real_cls_id
            if real_cls_id:
                statistics_dic[real_name+'_tpfn'] +=1
            else:
                statistics_dic[real_name+'_fptn'] +=1          
            if int(pred_cls_id) == int(real_cls_id)==1:
                statistics_dic[pred_name+'_tp'] +=1
                # print(pred_name,statistics_dic[pred_name+'_tp'])
            elif int(pred_cls_id)==int(real_cls_id)==0:
                statistics_dic[pred_name+'_tn'] +=1
            # display(img_data,idx)
        record_w.save(tmp_item)
    filewr.write("cls_name,tpr,fpr,precision,tp_fn,fp_tn,tp,fp\n")
    for key_name in properties:
        tp_fn = statistics_dic[key_name+'_tpfn']
        tp = statistics_dic[key_name+'_tp']
        tn = statistics_dic[key_name+'_tn']
        fp_tn = statistics_dic[key_name+'_fptn']
        fp = fp_tn - tn
        # print(key_name,tp_fn,fp_tn,tp,tn)
        tpr = float(tp) / tp_fn if tp_fn else 0.0
        fpr = float(fp) / fp_tn if fp_tn else 0.0
        precision = float(tp) / (tp+fp) if tp+fp else 0.0
        statistics_dic[key_name+'_tpr'] = tpr
        statistics_dic[key_name+'_fpr'] = fpr
        statistics_dic[key_name+'_P'] = precision
        # file_wr.write('>>> {} result is: tp_fn-{} | fp_tn-{} | tp-{} | fp-{}\n'.format(key_name,\
                    #    tp_fn,fp_tn,tp,fp))
        # file_wr.write('\t tpr:{:.4f} | fpr:{:.4f} | Precision:{:.4f}\n'.format(tpr,fpr,precision))
        filewr.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(key_name,tpr,fpr,precision,tp_fn,fp_tn,tp,fp))
    filerd.close()
    filewr.close()
    record_w.close()

def getF1(plist,rlist):
    '''
    plist: list of precision
    rlist: list of recall
    '''
    for tmpp,tmpr in zip(plist,rlist):
        f1 = float(tmpp*tmpr)/float(tmpp+tmpr)
        print("precison,recall,f1:",tmpp,tmpr,f1)

if __name__=='__main__':
    csv1='D:\Develop\jd_prj\mobilephone_recognition\data\\broken_p20_ecaresnet101d_result.csv'
    csv2='D:\Develop\jd_prj\mobilephone_recognition\data\\broken_p20_ecaresnet101d_result2.csv'
    outfile = '..\data\merge2csv.txt'
    # Merge2modelResult(csv1,csv2,outfile)
    csvfile = '..\data\\broken_p20_ecaresnet101d_result_merge.csv'
    recordfile = '..\data\\broken_p20_ecaresnet101d_merge_result.txt'
    # analysisfile(outfile,csvfile,recordfile)
    getF1([0.946,0.962,0.949,0.951,0.963,0.9199,0.9615,0.9513,0.9463],[0.985,0.976,0.98,0.983,0.976,0.988,0.9795,0.9866,0.9887]) #[0.117,0.194,0.157,0.137,0.197,0.09] mergep,v2,p1,p3,p32,p24