import os
import cv2
import hashlib
import numpy as np
import tqdm
import shutil

# md5编码文件
def encodeMd5(str):
    md = hashlib.md5()# 创建md5对象
    md.update(str.encode(encoding='utf-8'))
    return md.hexdigest()
 

def generateUnique(imgdir,fileout):
    fw = open(fileout,'w')
    file_cnts = os.listdir(imgdir)
    imgMD5List = []
    for i in tqdm.tqdm(range(len(file_cnts))):
        imgname = file_cnts[i].strip()
        imgpath = os.path.join(imgdir,imgname)
        img = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_COLOR)
        imgMd5 = encodeMd5(str(img))
        if imgMd5 not in imgMD5List:
            imgMD5List.append(imgMd5)
            fw.write(imgname+"\n")
    fw.close()

def generate_imglist(imgdir,outfile,level=1):
    '''
    imgdir: saved negtivate images
    outfile: [imagename]
    '''
    fcnts = os.listdir(imgdir)
    fw = open(outfile,'w')
    label = 0
    cnt = 0
    for tmp in fcnts:
        tmp = tmp.strip()
        if level==2:
            tmpdir = os.path.join(imgdir,tmp)
            t_fcnts = os.listdir(tmpdir)
            for j in tqdm.tqdm(range(len(t_fcnts))):
                tmpname = t_fcnts[j].strip()
                image_name = tmp +'\\'+tmpname
                fw.write("{}\n".format(image_name))
                cnt+=1
        else:
            fw.write("{}\n".format(tmp))
            cnt+=1
    fw.close()
    print(cnt)

def getIou(imgfile,md5file,fileout):
    fr1 = open(imgfile,'r')
    fr2 = open(md5file,'r')
    fw = open(fileout,'w')
    fcnts1 = fr1.readlines()
    fcnts2 = fr2.readlines()
    for i in tqdm.tqdm(range(len(fcnts1))):
        tmp = fcnts1[i].strip()
        tmp_spl = tmp.split('\\')[-1].split('_')
        imgname = '_'.join(tmp_spl[4:])
        for md5name in fcnts2:
            if imgname == md5name.strip():
                fw.write(tmp+'\n')
                break
    fr1.close()
    fr2.close()
    fw.close()

def copyimgsfromtxt(filein,imgdir,savedir):
    '''
    '''
    def build_dir(fpath):
        if not os.path.exists(fpath):
            os.makedirs(fpath)
    build_dir(savedir)
    # dir1 = os.path.join(savedir,'broken_imgs')
    # dir2 = os.path.join(savedir,)
    fr = open(filein,'r')
    fcnts = fr.readlines()
    tmpname = 'fg_img'
    cnt = 0
    for tmp in fcnts:
        # tmp_spl = tmp.strip().split('\\')
        # tmpname = tmp_spl[0]
        # tmpdir = os.path.join(savedir,tmpname)
        # build_dir(tmpdir)
        # cnt+=1
        # tmpid = int(cnt / 500)
        # tmpdir = os.path.join(savedir,tmpname+str(tmpid))
        # build_dir(tmpdir)
        orgpath = os.path.join(imgdir,tmp.strip())
        savepath = os.path.join(savedir,tmp.strip())
        if not os.path.exists(orgpath):
            continue
        shutil.move(orgpath,savepath)
    print("over")

if __name__=='__main__':
    # imgdir = 'D:\Datas\mobilephone\phone_pre_cls\\front_all\p20_front'
    imgdir = 'D:\Datas\mobilephone\phone_pre_cls\online\img_bg'
    # fileout1 = '..\data\p20_clear.txt'
    fileout1 = '..\data\online_bg.txt'
    generateUnique(imgdir,fileout1)
    # imgdir = 'D:\Datas\mobilephone\phone_detect\phone_crops\pall_crops\p20_crops'
    fileout2 = '..\data\p20_crops_org.txt'
    # generate_imglist(imgdir,fileout2,2)
    fileout3 = '..\data\p20_crops_clear.txt'
    # getIou(fileout2,fileout1,fileout3)
    # savedir = 'D:\Datas\mobilephone\phone_detect\phone_crops\pall_crops\p20_crops_clear'
    imgdir = 'D:\Datas\mobilephone\phone_pre_cls\online\img_bg'
    savedir = 'D:\Datas\mobilephone\phone_pre_cls\online\img_bg_clear'
    copyimgsfromtxt(fileout1,imgdir,savedir)
