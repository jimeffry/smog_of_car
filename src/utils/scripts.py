import numpy as np 
from matplotlib import pyplot as plt 
from collections import defaultdict
import os
import sys
import cv2
import tqdm
import shutil
import csv
import random

def get_txtfile3(imgdir,outfile1,outfile2,name,training=True):
    fw = open(outfile1,'w')
    if training:
        fw2 = open(outfile2,'w')
    datas = []
    f1_cnts = os.listdir(imgdir)
    # tmp_name = imgdir.split('/')[-1]
    cnt_dict = defaultdict(lambda:0)   
    for i in tqdm.tqdm(range(len(f1_cnts))):
        tmp_f = f1_cnts[i].strip()
        tmpdir = os.path.join(imgdir,tmp_f)
        if not os.path.isdir(tmpdir):
            continue
        if not (tmp_f in name):
            continue
        f2_cnts = os.listdir(tmpdir)
        for tmp_subdir in f2_cnts:
            #if tmp_subdir.strip() != 'front':
            #    continue
            tmp_subdir2 = os.path.join(tmpdir,tmp_subdir)
            image_names = os.listdir(tmp_subdir2)
            if 'unbroken' in tmp_subdir: #or 'hard' in tmp_subdir:
                label=0
            else:
                label=1
            tmp_total = len(image_names)
            key_num = np.ceil(tmp_total * 0.95)
            tmp_cnt = cnt_dict[tmp_subdir]
            cnt_dict[tmp_subdir]=tmp_cnt+tmp_total
            cnt_val = 0
            for imgname in image_names:
                cnt_val +=1
                tmppath = os.path.join(tmp_f+'\\'+tmp_subdir,imgname)
                #fw.write("{}\n".format(tmppath))
                if training:
                    if cnt_val < key_num:
                        fw.write("{},{}\n".format(tmppath,label))
                    else:
                        fw2.write("{},{}\n".format(tmppath,label))
                else:
                    fw.write("{},{}\n".format(tmppath,label))
                    # fw.write("{}\n".format(tmppath))
    # plothist(datas)
    fw.close()
    if training:
        fw2.close()
    keynames = cnt_dict.keys()
    for tmp in keynames:
        print("name - num:",tmp,cnt_dict[tmp])

def get_txtfile2(imgdir,outfile1,name):
    fw = open(outfile1,'w')
    # fw2 = open(outfile2,'w')
    datas = []
    f1_cnts = os.listdir(imgdir)
    # tmp_name = imgdir.split('/')[-1]
    for i in tqdm.tqdm(range(len(f1_cnts))):
        tmp_f = f1_cnts[i].strip()
        tmpdir = os.path.join(imgdir,tmp_f)
        if not (tmp_f in name):
            continue
        f2_cnts = os.listdir(tmpdir)
        if 'pos' in tmp_f:
            label = 2
        else:
            label = 0
        tmp_total = len(f2_cnts)
        key_num = np.ceil(tmp_total * 0.9)
        tmp_cnt = 0
        for imgname in f2_cnts:
            # imgpath = os.path.join(tmpdir,imgname.strip())
            # img = cv2.imread(imgpath)
            # h,w = img.shape[:2]
            # datas.append(min(h,w))
            tmp_cnt +=1
            tmppath = os.path.join(tmp_f,imgname)
            fw.write("{},{}\n".format(tmppath,label))
            # if tmp_cnt < key_num:
            #     fw.write("{},{}\n".format(imgname,label))
            # else:
            #     fw2.write("{},{}\n".format(imgname,label))
    # plothist(datas)
    fw.close()
    # fw2.close()

def get_txtfile1(imgdir,outfile):
    fw = open(outfile,'w')
    datas = []
    f1_cnts = os.listdir(imgdir)
    tmp_name = imgdir.split('\\')[-1]
    for i in tqdm.tqdm(range(len(f1_cnts))):
        tmp_f = f1_cnts[i].strip()
        tmppath = os.path.join(tmp_name,tmp_f)
        fw.write("{}\n".format(tmppath))


def image_preporcess(image, target_size):
    '''
    image: numpy_array
    target_size: img_w,img_h
    '''
    # resize 尺寸,199.232.69.194 140.82.113.4
    ih, iw = target_size
    # 原始图片尺寸
    h,  w, _ = image.shape
    # 计算缩放后图片尺寸
    scale = min(iw/w, ih/h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))
    # 创建一张画布，画布的尺寸就是目标尺寸 fill_value=128为灰色画布
    image_paded = np.full(shape=[ih, iw, 3], fill_value=128)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    # 将缩放后的图片放在画布中央
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    return image_paded.astype(np.uint8)

def saveAndresize2(imgdir,savedir):
    '''
    imgdir: dir1/dir2/imgs.png
    savedir:
    '''
    def mkdirs(ppath):
        if not os.path.exists(ppath):
            os.makedirs(ppath)
    
    f1_cnts = os.listdir(imgdir)
    id_cnt = 0
    for i in tqdm.tqdm(range(len(f1_cnts))):
        tmp_f = f1_cnts[i].strip()
        print(tmp_f)
        if tmp_f =='online_bg':
            continue
        tmpdir = os.path.join(imgdir,tmp_f)
        ssdir = os.path.join(savedir,tmp_f)
        mkdirs(ssdir)
        f2_cnts = os.listdir(tmpdir)
        for imgname in f2_cnts:
            id_cnt+=1
            imgpath = os.path.join(tmpdir,imgname.strip())
            img = cv2.imread(imgpath)
            if img is None:
                print(imgpath)
                continue
            # img = cv2.resize(img,(448,448))
            img = image_preporcess(img,(448,448))
            # str(id_cnt)+'.jpg'
            savepath = os.path.join(ssdir,imgname.strip())
            cv2.imwrite(savepath,img)

def saveAndresize1(imgdir,savedir):
    '''
    imgdir: dir1/dir2/imgs.png
    savedir:
    '''
    def mkdirs(ppath):
        if not os.path.exists(ppath):
            os.makedirs(ppath)
    
    f1_cnts = os.listdir(imgdir)
    id_cnt = 0
    mkdirs(savedir)
    for i in tqdm.tqdm(range(len(f1_cnts))):
        tmp_f = f1_cnts[i].strip()
        imgpath = os.path.join(imgdir,tmp_f)
        img = cv2.imread(imgpath)
        if img is None:
            print(imgpath)
            continue
        # img = cv2.resize(img,(448,448))
        img = image_preporcess(img,(448,448))
        id_cnt+=1
        # savepath = os.path.join(savedir,str(id_cnt)+'.jpg')
        savepath = os.path.join(savedir,tmp_f)
        cv2.imwrite(savepath,img)


def splitclassfromimg(imgdir,savedir):
    '''
    imgdir: images saved, image_name+bg_score.jpg
    '''
    fgdir = os.path.join(savedir,'fg')
    bgdir = os.path.join(savedir,'bg')
    if not os.path.exists(fgdir):
        os.makedirs(fgdir)
    if not os.path.exists(bgdir):
        os.makedirs(bgdir)
    fcnts = os.listdir(imgdir)
    for tmp in fcnts:
        tmpdir = os.path.join(imgdir,tmp.strip())
        tmpcnts = os.listdir(tmpdir)
        for imgname in tmpcnts:
            tmpsplit = imgname.strip().split('-')
            imgpath = os.path.join(tmpdir,imgname.strip())
            if float(tmpsplit[-1][:-4]) < 0.5:
                savepath = os.path.join(fgdir,imgname.strip())
            else:
                savepath = os.path.join(bgdir,imgname.strip())
            shutil.copyfile(imgpath,savepath)

def getcommondata(path1,path2):
    '''
    dataformat: filename, keys...
    return: data_dict
    '''
    def getcsvdata(filein,keyname):
        f_in = open(filein,'r')
        output = []
        reader = csv.DictReader(f_in)
        for f_item in reader:
            #print(f_item['filename'])
            output.append(f_item[keyname])
            # if f_item['state'] == 'valid':
            #     output.append(f_item[keyname])
        f_in.close()
        return output
    train_datas = getcsvdata(path1,'name')
    test_datas = []
    fr = open(path2,'r')
    fcnts = fr.readlines()
    for tmp in fcnts:
        tmp_s = tmp.strip().split(',')
        test_datas.append(tmp_s[0])
        # if int(tmp_s[-1])==0:
        #     test_datas.append(tmp_s[0])
    set1 = set(train_datas)
    set2 = set(test_datas)
    iouset = set1 & set2
    print("train data:",len(set1))
    print("test data:",len(set2))
    print(len(iouset))

def processResultCls(filein,imgdir,savedir,name):
    '''
    filein: imgpath,predict_label
    '''
    path1 = os.path.join(savedir,'background')
    path2 = os.path.join(savedir,'back')
    path3 = os.path.join(savedir,'front')
    path4 = os.path.join(savedir,'computer')
    path5 = os.path.join(savedir,'Ipad')
    path6 = os.path.join(savedir,'TV')
    apath1 = os.path.join(savedir,'broken')
    apath2 = os.path.join(savedir,'perfect')
    bpath1 = os.path.join(savedir,'unbroken_imgs')
    bpath2 = os.path.join(savedir,'broken_imgs')
    def build_dir(imgpath):
        if not os.path.exists(imgpath):
            os.makedirs(imgpath)
    # build_dir(path1)
    # build_dir(path2)
    # build_dir(path3)
    # build_dir(path4)
    # build_dir(path5)
    # build_dir(path6)
    # build_dir(apath2)
    # build_dir(apath1)
    build_dir(bpath1)
    build_dir(bpath2)
    # dir_list = [path1,path2,path3,path4,path5,path6]
    # dir_list = [path1,path2,path3]
    # dir_list = [apath2,apath1]
    dir_list = [bpath1,bpath2]
    fr = open(filein,'r')
    fcnts = fr.readlines()
    cnt = 0
    for i in tqdm.tqdm(range(len(fcnts))):
        cnt+=1
        tmp = fcnts[i].strip()
        tmp_s = tmp.split(',')
        tmpdir = dir_list[int(tmp_s[1])]
        orgpath = os.path.join(imgdir,tmp_s[0])
        name_spl = tmp_s[0].split('\\')
        # savename = name_spl[0]+'_'+tmp_s[-1]+'_'+name_spl[1]
        # orgpath = os.path.join(imgdir,name_spl[0]+'\\'+name_spl[1])
        # savename = tmp_s[0]
        # savename = 'p4_'+str(cnt)+'_'+tmp_s[0]
        savename = name+'_'+str(cnt)+'_'+name_spl[-1]
        # savename = name_spl[0]+'_'+str(cnt)+'_'+name_spl[-1]
        savepath = os.path.join(tmpdir,savename)
        shutil.copyfile(orgpath,savepath)
    fr.close()


def psnr(path1,path2):
    img1 = np.load(path1)
    # img1 = img1[:,:,::-1]
    img2 = np.load(path2)
    print(img1.shape,img2.shape)
    eucd = np.sqrt(np.sum((img1-img2)**2))
    print(eucd)

def getattributelabel(imgdir,outfile):
    '''
    imgdir: mobilephone images, part_light_broken_id.jpg
    ourfilr: part,bright,broken
    '''
    fw = open(outfile,'w')
    fcnts = os.listdir(imgdir)
    for tmp in fcnts:
        labels = []
        tmp = tmp.strip()
        if 'broken' in tmp:
            labels.append('1')
        else:
            labels.append('0')
        if 'bright' in tmp:
            labels.append('0')
        else:
            labels.append('1')
        if 'part' in tmp:
            labels.append('0')
        else:
            labels.append('1')
        
        label_str = ",".join(labels)
        fw.write("{},{}\n".format(tmp,label_str))

def randomcopy(imgdir,savedir,percent):
    '''
    '''
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    f1_cnts = os.listdir(imgdir)
    total_num = len(f1_cnts)
    print(total_num)
    idx_list = list(range(total_num))
    random.shuffle(idx_list)
    cnt = int(total_num * percent)
    idx_sub = random.sample(idx_list,cnt)
    for i in idx_sub:
        tmp_f = f1_cnts[i].strip()
        imgpath = os.path.join(imgdir,tmp_f)
        tmppath = os.path.join(savedir,tmp_f)
        shutil.copyfile(imgpath,tmppath)
    print(cnt)

def foremove1(imgdir,orgdir,disdir):
    '''
    imgdir: get image names 
    orgdir: being moved images
    disdir:  moving to dir
    '''
    fcnts = os.listdir(imgdir)
    # dirdict = {"bg0":'fg0',"bg1":'fg1',"bg2":'fg2',"bg3":'fg3'}
    dirdict = {"fg0":'bg0',"fg1":'bg1',"fg2":'bg2',"fg3":'bg3'}
    for tmp in fcnts:
        tmp_spl = tmp.strip().split('_')
        imgname = '_'.join(tmp_spl[2:])
        orgpath = os.path.join(orgdir,tmp_spl[0]+"\\"+imgname)
        tmpdir = dirdict[tmp_spl[0]]
        savepath = os.path.join(disdir,tmpdir+"\\"+imgname)
        if not os.path.exists(orgpath):
            print(orgpath)
            continue
        shutil.move(orgpath,savepath)
    print("over")
    
def foremove2(imgdir,orgdir,disdir):
    '''
    imgdir: get image names 
    orgdir: being moved images
    disdir:  moving to dir
    '''
    fcnts = os.listdir(imgdir)
    if not os.path.exists(disdir):
        os.makedirs(disdir)
    # dirdict = {"bg0":'fg0',"bg1":'fg1',"bg2":'fg2',"bg3":'fg3'}
    for tmp in fcnts:
        tmp_spl = tmp.strip().split('_')
        imgname = '_'.join(tmp_spl[5:])
        orgpath = os.path.join(orgdir,imgname)
        savepath = os.path.join(disdir,imgname)
        if not os.path.exists(orgpath):
            print("not exist:",orgpath)
            continue
        shutil.move(orgpath,savepath)
    print("over")

def removeimages(imgdir,disdir,name):
    '''
    imgdir: get image names 
    orgdir: being moved images
    disdir:  moving to dir
    '''
    fcnts = os.listdir(imgdir)
    # dirdict = {"bg0":'fg0',"bg1":'fg1',"bg2":'fg2',"bg3":'fg3'}
    d1 = 'unbroken_imgs'
    d2 = 'broken_imgs'
    dpath1 = os.path.join(disdir,d1)
    dpath2 = os.path.join(disdir,d2)
    def build_dir(imgpath):
        if not os.path.exists(imgpath):
            os.makedirs(imgpath)
    build_dir(dpath1)
    build_dir(dpath2)
    cnt = 0
    for tmp in fcnts:
        tmp = tmp.strip()
        if tmp == 'unbroken_imgs':
            tmpdir = dpath1
        else:
            tmpdir = dpath2
        subdir = os.path.join(imgdir,tmp)
        sub_cnts = os.listdir(subdir)
        for subtmp in sub_cnts:
            subtmp = subtmp.strip()
            imgname = subtmp #'_'.join(tmp_spl[3:])
            orgpath = os.path.join(subdir,imgname)
            savename = name+'_'+str(cnt)+'_'+imgname
            savepath = os.path.join(tmpdir,savename)
            if not os.path.exists(orgpath):
                print(orgpath)
                continue
            shutil.move(orgpath,savepath)
            cnt+=1
    print("over")

def copyimgs(imgdir,savedir):
    '''
    imgdir: pnum_idnum_name.jpg
    '''
    fcnts = os.listdir(imgdir)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    cnt = 0
    for i in tqdm.tqdm(range(len(fcnts))):
        tmp = fcnts[i]
        # tmp_spl = tmp.strip().split('_')
        # imgname = '_'.join(tmp_spl[2:])
        cnt +=1
        imgname = 'card_'+str(cnt)+'_'+tmp.strip()
        orgpath = os.path.join(imgdir,tmp.strip())
        img = cv2.imread(orgpath)
        img = cv2.resize(img,(448,448))
        savepath = os.path.join(savedir,imgname)
        if not os.path.exists(orgpath):
            print(orgpath)
            continue
        # shutil.copyfile(orgpath,savepath)
        cv2.imwrite(savepath,img)
    print("over")

def copyimagesfromtxt(filein,imgdir,savedir):
    '''
    filein: sundir/pnum_idx_name.jpg
    '''
    fr = open(filein,'r')
    fcnts = fr.readlines()
    # dir1 = os.path.join(savedir,'unbroken_imgs')
    # dir2 = os.path.join(savedir,'broken_imgs')
    def build_dir(imgpath):
        if not os.path.exists(imgpath):
            os.makedirs(imgpath)
    # build_dir(dir1)
    # build_dir(dir2)
    cnt = 0
    pc = 0
    pn=0
    # orgname_dict = {"p20_crops_clear":"p2-0","p21_crops":"p2-1","p22_crops":"p2-2",\
        # "p23_crops":"p2-3","p24_crops":"p2-4","p25_crops_clear":"p2-5","wyk_crops":"wyk"}
    for i in tqdm.tqdm(range(len(fcnts))):
        tmp = fcnts[i].strip()
        tmp_spl = tmp.split(',')
        tmp_path_spl = tmp_spl[0].split('\\')
        # if "wyk_crops" != tmp_path_spl[0]:
        #     pc+=1
        #     continue
        tmpdir = os.path.join(savedir,tmp_path_spl[0],tmp_path_spl[1])
        build_dir(tmpdir)
        # tmp_name_spl = tmp_spl[1].split('_')
        # imgname = '_'.join(tmp_name_spl[2:])
        # tmp_name_spl = tmp_path_spl[-1].split('_')
        # imgname = "_".join(tmp_name_spl[2:])
        # orgname = orgname_dict[tmp_path_spl[0]]
        imgname = tmp_spl[0]
        savename = 'fpr'+'_'+tmp_spl[1]+'_'+tmp_spl[2]+'_'+tmp_path_spl[-1]
        orgpath = os.path.join(imgdir,imgname)
        savepath = os.path.join(tmpdir,savename)
        if os.path.exists(savepath) :
            continue
        elif not os.path.exists(orgpath):
            pn+=1
            continue
        shutil.copyfile(orgpath,savepath)
        cnt+=1
    print(cnt)
    print(pc)
    print(pn)

def getfprdata(filein,fileout):
    '''
    dataformat: filename, keys...
    return: data_dict
    '''
    f_in = open(filein,'r')
    fw = open(fileout,'w')
    reader = csv.DictReader(f_in)
    for f_item in reader:
        #print(f_item['filename'])
        cur_data = []
        imgpath = f_item['filename']
        # name_spl = imgpath.split('/')
        if 'unbroken' in imgpath:
            label = 0
        else:
            label = 1
        for cur_key in ['unbroken','broken']:
            cur_data.append(float(f_item[cur_key]))
        pred_id = np.argmax(np.array(cur_data))
        # if pred_id ==2:
        #     tmp_score = cur_data[pred_id]
        #     if tmp_score< 0.8:
        #         cur_data[pred_id] = 0.0
        #         pred_id = np.argmax(cur_data)
        if int(pred_id) != label:
            fw.write("{},{},{}\n".format(imgpath,pred_id,cur_data[pred_id]))
    f_in.close()
    fw.close()

def process_bboxes(filein,imgdir,savedir):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    data_r = open(filein,'r')
    annotations = data_r.readlines()
    bbox_list = []
    print("begin to process datas:")
    for i in tqdm.tqdm(range(len(annotations))):
        tmp = annotations[i].strip()
        tmp_splits = tmp.split(',')
        # img_path = os.path.join(imgdir,tmp_splits[0])
        # img_name = tmp_splits[0].split('/')[-1][:-4] if len(tmp_splits[0].split('/')) >0 else tmp_splits[0][:-4]
        if len(tmp_splits) >=2 :
            bbox = map(float, tmp_splits[1:])
            if not isinstance(bbox,list):
                bbox = list(bbox)
            imgname = tmp_splits[0]
            imgpath = os.path.join(imgdir,tmp_splits[0])
            if not os.path.exists(imgpath):
                continue
            img = cv2.imread(imgpath)
            imgh,imgw,_ = img.shape
            bbox = np.array(bbox).reshape([-1,5])
            bbox = bbox[:,:4]
            bbox = np.maximum(bbox,0)
            bbox[:,::2] = np.minimum(bbox[:,::2],imgw-1)
            bbox[:,1::2] = np.minimum(bbox[:,1::2],imgh-1)
            # bh = bbox[:,3]-bbox[:,1]
            # bw = bbox[:,2]-bbox[:,0]
            # bw,bh = bw/imgw * nw, bh/imgh * nh
            # bbox_list.extend(np.sqrt(bh*bw).tolist())
            cnt = 0 
            for tmpbox in bbox:
                x1,y1,x2,y2 = tmpbox.astype(np.int32)
                w = x2-x1
                h = y2-y1
                if w>5 and h >5:
                    tmpimg = img[y1:y2,x1:x2,:]
                    tmpname = 'crop'+'_'+str(cnt)+'_'+imgname
                    cnt+=1
                    # print(tmpbox)
                    savepath = os.path.join(savedir,tmpname)
                    if os.path.exists(savepath):
                        continue
                    cv2.imwrite(savepath,tmpimg)
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),4) 
            # img = cv2.resize(img,(640,640))
            # cv2.imshow('src',img)
            # cv2.waitKey(0)      

def afterProcess(imgdir,orgdir,savedir):
    fcnts = os.listdir(imgdir)
    for tmp in fcnts:
        tmp_spl = tmp.strip().split('_')
        imgname = '_'.join(tmp_spl[6:])
        orgpath = os.path.join(orgdir,imgname)
        savepath = os.path.join(savedir,imgname)
        if os.path.exists(orgpath):
            shutil.move(orgpath,savepath)
        else:
            print(orgpath)
    print('over')


if __name__=='__main__':
    # get_txtfile1('E:\datas_anlian\pre_filter_datas_train\\train_datas\p2-0','..\data\p2_0.txt')
    # saveAndresize('D:\Datas\p2_2','D:\Datas\p2_2_rp')
    # get_txtfile2('/home/lixiaoyu80/Datas/p2_1_rp','../data/test_p21.txt')
    # get_txtfile2('D:\Datas\p2_2','..\data\\test_data2.txt')
    # getcommondata('..\data\pre_filter_data.csv','..\data\\test_data2.txt')
    # imgdir = 'D:\Datas\mobilephone\phone_pre_cls\online\data'
    imgdir = 'D:\Datas\mobilephone\phone_detect\phone_crops\pall_crops'
    # imgdir = 'D:\Datas\mobilephone\phone_pre_cls\quality_imgs'
    # get_txtfile2(imgdir,'..\data\online_test.txt',['pos_samples','neg_samples'])
    # get_txtfile3(imgdir,'..\data\\blur_crops.txt',None,['p20_crops_clear','p25_crops_clear','p21_crops','p22_crops','p23_crops','p24_crops'],False)
    # get_txtfile3(imgdir,'..\data\\p2025plus_crops.txt',None,['p20_crops_clear','p25_crops_clear'],False)
    #processResultCls('..\data\\res50_fpr.txt','D:\Datas\p2_2_rp','D:\Datas\p22_result')
    # saveAndresize1('D:\Datas\\test_online\\onlinev1\online_datas','D:\Datas\\test_online\\online_rp')
    # processResultCls('..\data\\res50_fpr_p21.txt','D:\Datas\p2_1_rp','D:\Datas\p21_result')
    #processResultCls('..\data\\res50_fpr_p21.txt','D:\Datas\p2_1_rp','D:\Datas\p21_result')
    #processResultCls('..\data\\res50_fpr_p23.txt','D:\Datas\p2_3_rp','D:\Datas\p23_result')
    # lixiaoyu('D:\Datas\lixiaoyu.jpg','D:\Datas\lixiaoyu2.jpg')
    #processResultCls("..\data\\res50v2_fpr_p20.txt","D:\Datas\mobilephone\p2_0_rp","D:\Datas\mobilephone\p20_res50v2_result")
    #processResultCls("..\data\\result_online.txt","D:\Datas\\test_online\online_rp","D:\Datas\\test_online\online_cls")
    # processResultCls("..\data\\res50_fpr_online.txt","D:\Datas\\test_online\online_cls","D:\Datas\\test_online\online_result")
    # lixiaoyu("D:\Datas\\test_online\online_rp\\1.jpg","")
    # plothist([])
    #saveAndresize1("D:\Datas\mobilephone\phone_pre_cls\p2_5","D:\Datas\mobilephone\phone_pre_cls\p25_tmp") 
    # psnr("D:\Develop\jd_prj\mobilephone_recognition\src\\tmp\8ed137f8.jpg.npy","D:\Develop\jd_prj\ins-ai-vision-mq\\fght_mq\8ed137f8.jpg.npy")
    # getattributelabel("D:\Datas\mobilephone\\rename_data_wyk1205\wyk","..\data\wyk_data.txt")
    # processResultCls("..\data\phonetest_result.txt","D:\Datas\mobilephone\phone_pre_cls","D:\Datas\mobilephone\phone_attribute")
    # saveAndresize1("D:\Datas\mobilephone\phone_pre_cls\p25_result\\front","D:\Datas\mobilephone\phone_pre_cls\\front2broken\p25f")
    imgdir = 'D:\Datas\mobilephone\phone_detect\phone_crops\pall_crops\p21_front'
    savedir = 'D:\Datas\mobilephone\phone_detect\phone_crops\pall_crops\p21_crops'
    # processResultCls("..\data\p21_broken_result.txt",imgdir,savedir)
    # randomcopy("\Datas\\test_online\onlinev5\onlinev5_result\\back","\Datas\\test_online\onlinev5\onlinev5_label\\back",0.3)
    imgdir = "D:\Datas\mobilephone\phone_pre_cls\p2_0"
    savedir = "D:\Datas\mobilephone\phone_pre_cls\p20_result"
    # processResultCls("..\data\p20_tmp_result.txt",imgdir,savedir)
    imgdir ='D:\Datas\mobilephone\phone_detect\phone_crops\pall_crops'
    savedir = 'D:\Datas\mobilephone\phone_detect\phone_crops\pall_crops\p25_crops'
    # processResultCls("..\data\\p25_broken_result.txt",imgdir,savedir,'p5')
    imgdir = "D:\Datas\mobilephone\phone_attribute\p24_front\\bg4"
    savedir = "D:\Datas\mobilephone\phone_attribute\p24_front\\fg4"
    # foremove2("D:\Datas\mobilephone\phone_attribute\p24_ecaresnet101_result\\bg2fg",imgdir,savedir)
    imgdir = "D:\Datas\mobilephone\phone_attribute"
    savedir = "D:\Datas\mobilephone\phone_attribute"
    #foremove("D:\Datas\mobilephone\phone_attribute\\train_test_result\\fg2bg",imgdir,savedir)
    savedir = "D:\Datas\mobilephone\phone_pre_cls\\front2broken\p24f"
    # copyimgs("D:\Datas\mobilephone\phone_pre_cls\p24_result\\front",savedir)
    imgdir = 'D:\Datas\mobilephone\phone_pre_cls\p2_4'
    savedir = 'D:\Datas\mobilephone\phone_attribute\p24_org'
    # copyimagesfromtxt("..\data\\broken_p24_test.txt",imgdir,savedir)
    # getfprdata('..\data\\p2025plus_broken_result.csv','..\data\p2025plus_crops_fpr.txt')
    imgdir = 'D:\Datas\mobilephone\phone_detect\phone_crops\pall_crops'
    savedir = 'D:\Datas\\broken_p3_test_result'
    # copyimagesfromtxt('..\data\\p2025plus_crops_fpr.txt',imgdir,savedir)
    imgdir = 'D:\JDNetDiskDownload\\noBroken_oriPic'
    savedir = 'D:\JDNetDiskDownload\\noBroken_oriPic_delete'
    foremove2('D:\Datas\\broken_p3_test_result\p20_crops_clear\\0518_2fg',imgdir,savedir)
    filein = '..\data\\broken_boxes.txt'
    imgdir = 'D:\Datas\mobilephone\\rename_data_wyk1205\wyk'
    savedir = 'D:\Datas\mobilephone\phone_detect\phone_crops\\broken_crops\\broken'
    # process_bboxes(filein,imgdir,savedir)
    imgdir = 'D:\Datas\mobilephone\phone_detect\phone_crops\pall_crops\p24_crops'
    savedir = 'D:\Datas\mobilephone\phone_detect\phone_crops\pall_crops\p24_crops_rename'
    # removeimages(imgdir,savedir,'p4')
    imgdir='D:\Datas\mobilephone\phone_detect\phone_crops\pall_crops\p2025_resultp3\\false\\fg2bg_unbroken'
    orgdir = 'D:\Datas\mobilephone\phone_detect\phone_crops\pall_crops\p25_crops_clear\\fg2bg_unbroken'
    savedir = 'D:\Datas\mobilephone\phone_detect\phone_crops\pall_crops\p25_crops_clear\\broken_imgs'
    # afterProcess(imgdir,orgdir,savedir)
    imgdir = 'D:\Datas\mobilephone\phone_pre_cls\p2_3\\background'
    savedir = 'D:\Datas\mobilephone\phone_pre_cls\p2_3\\background_copy'
    # copyimgs(imgdir,savedir)
    imgdir = "E:\datas_anlian\\20万源数据\datas"
    imgdir = "D:\Datas\mobilephone\\rename_data_wyk1205"
    savedir = "F:\datas_anlian\\attribute_broken"
    # copyimagesfromtxt("..\data\\broken_all_data.txt",imgdir,savedir)