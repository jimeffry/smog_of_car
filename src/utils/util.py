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
import errno
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

def download_file(url, local_fname=None, force_write=False):
    # requests is not default installed
    import requests
    if local_fname is None:
        local_fname = url.split('/')[-1]
    if not force_write and os.path.exists(local_fname):
        return local_fname

    dir_name = os.path.dirname(local_fname)

    if dir_name != "":
        if not os.path.exists(dir_name):
            try: # try to create the directory if it doesn't exists
                os.makedirs(dir_name)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

    r = requests.get(url, stream=True)
    assert r.status_code == 200, "failed to open %s" % url
    with open(local_fname, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
    return local_fname

def get_gpus():
    """
    return a list of GPUs
    """
    try:
        re = subprocess.check_output(["nvidia-smi", "-L"], universal_newlines=True)
    except OSError:
        return []
    return range(len([i for i in re.split('\n') if 'GPU' in i]))

def get_by_ratio(x,new_x,y):
    ratio = x / float(new_x)
    new_y = y / ratio
    return np.floor(new_y)


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
    
    def parse(self,datadir):
        cnts = os.listdir(datadir)
        smog_cnt = 0
        failed = []
        for tmp in cnts:
            tmp_file = os.path.join(datadir,tmp.strip())
            fg = self.getXmlData(tmp_file)
            if fg ==1:
                smog_cnt+=1
            else:
                failed.append(tmp_file)
        print(smog_cnt)
        print(failed)

    
if __name__ == '__main__':
    args = parms()
    file_in = args.file_in
    file_out = args.out_file
    cmd = args.cmd_type
    file_in = "d:\homeLxy\datas\汽车\训练数据\Annotations"
    if cmd=='gen_trainfile':
        generate_train_label(file_in,file_out)
    else:
        xml_p = XmlParse()
        xml_p.parse(file_in)
        print("please input cmd right")