# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import subprocess
import os
import numpy as np 
import cv2
import argparse
import csv
import xml.etree.ElementTree as ET

def parms():
    parser = argparse.ArgumentParser(description='gen ')
    parser.add_argument('--img-dir',type=str,dest="img_dir",default='./',\
                        help='the directory should include 2 more Picturess')
    parser.add_argument('--file-in',type=str,dest="file_in",default="train.txt",\
                        help='img paths saved file')
    parser.add_argument('--save-dir',type=str,dest="save_dir",default='./',\
                        help='img saved dir')
    parser.add_argument('--base-id',type=int,dest="base_id",default=0,\
                        help='img id')
    parser.add_argument('--out-file',type=str,dest="out_file",default="train.txt",\
                        help='out img paths saved file')
    parser.add_argument('--cmd-type',type=str,dest="cmd_type",default="None",\
                        help='which code to run: gen_trainfile')
    parser.add_argument('--file2-in',type=str,dest="file2_in",default="train2.txt",\
                        help='label files')
    return parser.parse_args()


def generate_train_label(file_in,fileout):
    '''
    file_in: input_label csv file
    fileout: ouput train file
    '''
    f_in = open(file_in,'rb')
    dict_keys = f_in.readline().strip().split('')
    f_in.close()
    f_in = open(file_in,'rb')
    print('read data dict_keys:',dict_keys)
    list_out = []
    for name in dict_keys:
        list_out.append([])
    data_dict = dict(zip(dict_keys,list_out))
    reader = csv.DictReader(f_in)
    print(len(reader))
    #for f_item in reader:
        #print(f_item['filename'])
    f_in.close()

class XmlParse(object):
    def __init__(self) -> None:
        pass

    def getXmlData(self,xmlFile):
        '''
        '''
        target = ET.parse(xmlFile).getroot()
        fg = 0
        for obj in target.iter("object"):
            name = obj.find('name').text.lower().strip()
            if name == "smog":
                fg = 1
        return fg
    
    def parse(self,datadir,fileout):
        recored = open(fileout,'w')
        cnts = os.listdir(datadir)
        smog_cnt = 0
        failed = 0
        for tmp in cnts:
            tmp_file = os.path.join(datadir,tmp.strip())
            fg = self.getXmlData(tmp_file)
            if fg ==1:
                smog_cnt +=1
                recored.write(tmp.strip()+"\n")
            else:
                failed +=1
        print("total data:",len(cnts))
        print(smog_cnt)
        print(failed)
        recored.close()

class GenLabel(object):
    def __init__(self) -> None:
        self.label0 = "no_smog"
        self.label1 = "smog"
        self.label2 = "yisi"

    def parse(self,imgdir,tp_file,save_dir):
        t_file = os.path.join(save_dir,"train.txt")
        v_file = os.path.join(save_dir,"val.txt")
        f_train = open(t_file,"w")
        f_val = open(v_file,"w")
        fr = open(tp_file,"r")
        cnts = fr.readlines()
        tp_list = []
        for tmp in cnts:
            tp_list.append(tmp.strip().split(".")[0])
        img_names = os.listdir(imgdir)
        pos_cnt = 0
        neg_cnt = 0
        for tmp in img_names:
            tmp = tmp.strip()
            tmp_spl = tmp.split(".")
            label = 0
            if tmp_spl[0] in tp_list:
                label = 1
            if label==1:
                pos_cnt+=1
            else:
                neg_cnt+=1
            if label==1 and pos_cnt <=1000:
                f_val.write("{},{}\n".format(tmp,label))
            elif label==0 and neg_cnt <=500:
                f_val.write("{},{}\n".format(tmp,label))
            else:
                f_train.write("{},{}\n".format(tmp,label))
        print("pose:",pos_cnt)
        print("neg:",neg_cnt)

    def genNeg(self,imgdir,save_dir):
        neg_file = os.path.join(save_dir,"neg.txt")
        f_val = open(neg_file,"w")
        img_names = os.listdir(imgdir)
        neg_cnt = 0
        for tmp in img_names:
            tmp = tmp.strip()
            f_val.write("{},{}\n".format(tmp,0))
            neg_cnt +=1
        print("neg:",neg_cnt)

    def genalllabel(self,image_dir,save_dir):
        train_file = os.path.join(save_dir,"train_3cls.txt")
        val_file = os.path.join(save_dir,"val_3cls.txt")
        fw_train = open(train_file,"w")
        fw_val = open(val_file,"w")
        cnts = os.listdir(image_dir)
        val_cnt = dict()
        for tmp in cnts:
            tmp = tmp.strip()
            if tmp == self.label0:
                label = 0
            elif tmp == self.label1:
                label = 1
            elif tmp == self.label2:
                label = 2
            else:
                print("not valid image dir:",tmp)
            val_cnt[label] = 0
            tmp_dir = os.path.join(image_dir,tmp)
            if os.path.isdir(tmp_dir):
                tmp_cnts = os.listdir(tmp_dir)
                total_num = len(tmp_cnts)
                for imgname in tmp_cnts:
                    tmp_name = os.path.join(tmp,imgname.strip())
                    val_cnt[label] = val_cnt[label]+1
                    if val_cnt[label] <= int(total_num*0.1):
                        fw_val.write("{},{}\n".format(tmp_name,label))
                    else:
                        fw_train.write("{},{}\n".format(tmp_name,label))
            print(val_cnt[label])
        print("over")



    
if __name__ == '__main__':
    args = parms()
    file_in = args.file_in
    file_out = args.out_file
    cmd = args.cmd_type
    file_in = "/home/lixy/datas/smog_car/annotations"
    file_out = "/home/lixy/datas/smog_car/"
    if cmd=='gen_trainfile':
        generate_train_label(file_in,file_out)
    else:
        # fileout = os.path.join(file_out,"record_smog.txt")
        # xml_p = XmlParse()
        # xml_p.parse(file_in,fileout)
        # print("please input cmd right")
        #*
        gt = GenLabel()
        # #imgdir = "/home/lixy/datas/smog_car/images"
        # labelfile = "/home/lixy/datas/smog_car/record_smog.txt"
        # sdir = "/home/lixy/datas/smog_car"
        # gt.parse(imgdir,labelfile,sdir)
        #**
        # imgdir = "/home/lixy/datas/smog_car/infer_result/smog"
        # sdir = "/home/lixy/datas/smog_car/infer_result"
        # gt.genNeg(imgdir,sdir)
        imgdir = "/home/lg/图片/黑烟整理素材"
        sdir = "/home/lg/Project/lixy/smog_of_car/data"
        gt.genalllabel(imgdir,sdir)