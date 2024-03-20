import sys
import os
import argparse
import shutil
import json

import cv2
import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from load_model import MobilePhoneNetTR,MobilePhoneNetTF,PhoneAttributeTF
sys.path.append(os.path.join(os.path.dirname(__file__),'..\configs'))
from config import cfg



def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
def parms():
    parser = argparse.ArgumentParser(description='CSRnet demo')
    parser.add_argument('--save_dir', type=str, default='tmp/',
                        help='Directory for detect result')
    parser.add_argument('--modelpath', type=str,
                        default='d:\models\phone_cls\\broken_crops_p3_ecaresnet101d.pth', help='trained model')
    parser.add_argument('--threshold', default=0.65, type=float,
                        help='Final confidence threshold')
    parser.add_argument('--cuda_use', default=True, type=str2bool,
                        help='gpu run')
    parser.add_argument('--file_in', type=str, default=None,
                        help='image namesf')
    parser.add_argument('--file_out', type=str, default=None,help='result output files')
    parser.add_argument('--minsize',type=int,default=24,\
                        help="scale img size")
    parser.add_argument('--img-dir',type=str,dest='img_dir',default="./",\
                        help="images saved dir")
    parser.add_argument('--base-name',type=str,dest='base_name',default="videox",\
                        help="images saved dir")
    parser.add_argument('--save-dir2',type=str,dest='save_dir2',default=None,\
                        help="images saved dir")
    parser.add_argument('--crop-size',type=str,dest='crop_size',default='112,112',\
                        help="images saved size")
    parser.add_argument('--detect_modelpath',type=str,default=None,help='retinateface')
    parser.add_argument('--detect-model-dir',type=str,dest='detect_model_dir',default="../../models/",\
                        help="models saved dir")
    return parser.parse_args()


class ModelTest(object):
    def __init__(self,args):
        self.Model = MobilePhoneNetTR(args)
        # self.Model = PhoneAttributeTF(args)
        # self.Model.register()
        print("***************load model over")
        self.threshold = args.threshold
        self.kernel_size = 5
        self.img_dir = args.img_dir
        self.real_num = 0
        self.save_dir = args.save_dir
        if self.save_dir is not None :
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        self.file_out = args.file_out
    
    
    def label_show(self,img,scores,pred_id):
        '''
        scores: shape-[batch,cls_nums]
        pred_id: shape-[batch,cls_nums]
        rectangles: shape-[batch,15]
            0-5: x1,y1,x2,y2,score
        '''
        show_labels = ['normal','unsell']
        colors = [(0,0,255),(255,0,0)]
        tmp_pred = pred_id
        tmp_score = '%.2f' % scores[int(tmp_pred)]
        show_name = show_labels[int(tmp_pred)] #+'_'+tmp_score
        color = colors[int(tmp_pred)]
        font=cv2.FONT_HERSHEY_COMPLEX_SMALL
        # font_scale = int((box[3]-box[1])*0.01)
        points = (10,10)
        cv2.putText(img, show_name, points, font, 1.0, color, 2)
        return img
        
    def inference_img(self,img):
        t1 = time.time()
        # imgorg = img.copy()
        # orgreg = 0
        # pred_ids = []
        scores,pred_ids = self.Model.inference([img])
        # img_out = self.label_show(img,scores[0],pred_ids[0])
        #print('consuming:',t2-t1)
        return scores[0],pred_ids

    def putrecord(self,datalist):
        file_w = open('ehualu_test.json','w',encoding='utf-8')
        fp = json.dump(datalist,ensure_ascii=False)
        file_w.write(fp)
        file_w.close()

    def resize_scale(self,image, target_size):
        '''
        image: numpy_array
        target_size: img_w,img_h
        '''
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

    def get_boxarea(self,hotmap,frame):
        '''
        hotmap: gray img
        '''
        imgh,imgw = hotmap.shape[:2]
        # mean_value = hotmap/float(imgh*imgw) 
        # mean_value = mean_value.sum()
        # print('min',np.min(img))
        # print('max',np.max(img))
        # print("mean:",mean_value)
        # img = np.where(img >0.0002,255,0)
        hot_list = np.reshape(hotmap,[-1])
        indx = np.argsort(hot_list)[::-1]
        top5 = int(0.05*imgw*imgh)
        top5_value = hot_list[indx[top5]]
        if top5 >= 120:
            threshold = 120 #mean_value
        else:
            threshold = top5_value
        # threshold = 120 #mean_value
        _,hotmap = cv2.threshold(hotmap,threshold,255,cv2.THRESH_BINARY)
        hotmap = np.array(hotmap,dtype=np.uint8)
        # cv2.imshow('thresh',img)
        kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size,1 ))
        kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, self.kernel_size))
        hotmap = cv2.dilate(hotmap, kernelX, iterations=2)
        hotmap = cv2.erode(hotmap, kernelX,  iterations=4)
        hotmap = cv2.dilate(hotmap, kernelY,  iterations=2)
        hotmap = cv2.erode(hotmap, kernelY,  iterations=1)
        hotmap = cv2.dilate(hotmap, kernelY,  iterations=2)
        hotmap = cv2.medianBlur(hotmap, 3)
        # cv2.imshow('dilate&erode', hotmap)
        # image, contours, hier = cv2.findContours(hotmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(hotmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        hull = []
        for i ,c in enumerate(contours):
            # 边界框
            x, y, w, h = cv2.boundingRect(c)
            hull.append(cv2.convexHull(c, False))
            if min(w,h) > 1:
                x1,x2,y1,y2 = int(x),int(x+w),int(y),int(y+h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                boxes.append([x1,y1,x2,y2])
                # cv2.polylines(frame, [hull[i]], True, (0, 255, 0), 2)
            # cv2.drawContours(frame, hull, i, (0, 255, 0), 1, 8)
            # cv2.polylines(frame, [hull[i]], True, (0, 255, 0), 2)
        return frame,boxes

    def __call__(self,filein):
        if os.path.isdir(filein):
            cnts = os.listdir(filein)
            for tmp in cnts:
                tmppath = os.path.join(filein,tmp.strip())
                # img = np.load(tmppath)
                # img = cv2.imdecode(np.fromfile(tmppath, dtype=np.uint8), -1)
                img = cv2.imdecode(np.fromfile(tmppath, dtype=np.uint8), cv2.IMREAD_COLOR)
                # img = cv2.imread(tmppath)
                # h,w = img.shape[:2]
                # maxl = max(h,w)
                # img_encode = cv2.imencode('.jpg', img)[1]
                # data_encode = np.array(img_encode)
                # str_encode = data_encode.tostring()
                # image = np.asarray(bytearray(str_encode), dtype="uint8")
                # img = cv2.imdecode(image, cv2.IMREAD_COLOR)
                # np.save('tmp/'+tmp.strip()+'.npy',img)
                if img is None:
                    print("img is none:",tmppath)
                    continue
                score,pids = self.inference_img(img.copy())
                pid = pids[0]
                print("path, scores and cls_id:",score,pid)
                # cv2.imshow('result',img)
                # img = self.resize_scale(img,(maxl,maxl))
                # img = cv2.resize(img,(cfg.InputSize_w,cfg.InputSize_h))
                cams = self.Model.getCam(img,pids)
                # names=["normal","broken"]
                txt = "%s_%.3f" % (str(pid), score[int(pid)])#[int(pid)]
                save_name = 'cam_'+txt+'_'+tmp.strip()[:-4]+'.jpg'
                # cv2.putText(img, txt, (100,100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4.0, (255,0,0), 8)
                savepath = os.path.join(self.save_dir,save_name)
                # cv2.imwrite(savepath,img)
                showimg,_ = self.get_boxarea(cams[0][0],np.array(cams[0][1],dtype=np.uint8))
                cv2.imwrite(savepath,showimg)
                # cv2.imshow('cam',np.array(cams[0][1],dtype=np.uint8))
                # cv2.imshow('box',showimg)
                # cv2.waitKey(0) 
        elif os.path.isfile(filein) and filein.endswith('txt'):
            f_r = open(filein,'r')
            file_cnts = f_r.readlines()
            #**********record
            label_name_list = ['valid','back','invalid']
            data_list = []
            fw = open(self.file_out,'w')
            #******** for label and to be classed
            # bg_path = os.path.join(self.save_dir,'invalid')
            # fg2_path = os.path.join(self.save_dir,'valid')
            # fg1_path = os.path.join(self.save_dir,'back')
            # pathlist = [fg2_path,fg1_path,bg_path]
            #
            # if not os.path.exists(bg_path):
            #     os.makedirs(bg_path)
            # if not os.path.exists(fg1_path):
            #     os.makedirs(fg1_path)
            # if not os.path.exists(fg2_path):
            #     os.makedirs(fg2_path)
            total_num = len(file_cnts)
            for j in tqdm(range(total_num)):
                # sys.stdout.write(">\r %d/%d" %(j,total_num))
                # sys.stdout.flush()
                # record_dict = dict()
                tmp_file = file_cnts[j].strip()
                #*************** here is for label file test
                tmp_file_s = tmp_file.split(',')
                # tmp_name = tmp_file_s[-1]
                if len(tmp_file_s)>1:
                    tmp_file = tmp_file_s[0]
                    real_label = int(tmp_file_s[1])
                # # if not tmp_file.endswith('jpg'):
                #     # tmp_file = tmp_file +'.jpg'
                tmp_path = os.path.join(self.img_dir,tmp_file) 
                # tmp_path = tmp_file
                if not os.path.exists(tmp_path):
                    print(tmp_path)
                    continue
                img = cv2.imread(tmp_path) 
                if img is None:
                    print('None',tmp_path)
                    continue
                scores,pred_id = self.inference_img(img)
                fw.write("{},{}\n".format(tmp_file,pred_id))
                #************ smoking and calling test
                # record_dict['image_name'] = tmp_file
                # record_dict['category'] = label_name_list[int(pred_id)]
                # record_dict['score'] = '%.5f' % scores[int(pred_id)]
                # data_list.append(record_dict)
                #************to be classed images
                # dist_path = os.path.join(pathlist[pred_id],tmp_name)
                # shutil.copyfile(tmp_path,dist_path)
                #********* label test files
                #if int(pred_id) != real_label:
                    # tmp_file_s = tmp_file.split('/')
                 #   fw.write("{},{}\n".format(tmp_file,pred_id))
                #     dist_path = os.path.join(pathlist[real_label],tmp_file_s[-1])
                #     shutil.copyfile(tmp_path,dist_path)
            # self.putrecord(data_list)
        elif os.path.isfile(filein) and filein.endswith(('.mp4','.avi')) :
            cap = cv2.VideoCapture(filein)
            if not cap.isOpened():
                print("failed open camera")
                return 0
            else: 
                while cap.isOpened():
                    _,img = cap.read()
                    frame,cnt_head = self.inference_img(img)
                    cv2.imshow('result',frame)
                    q=cv2.waitKey(10) & 0xFF
                    if q == 27 or q ==ord('q'):
                        break
            cap.release()
            cv2.destroyAllWindows()
        elif os.path.isfile(filein):
            img = cv2.imread(filein)
            if img is not None:
                # grab next frame
                # update FPS counter
                score,pid = self.inference_img(img)
                print("scores and cls_id:",score,pid)
                img = self.Model.resize_scale(img,(224,224))
                cv2.imshow('img',img)
                # cv2.imwrite('test_a1.jpg',frame)
                key = cv2.waitKey(0) 
        elif filein=='video':
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("failed open camera")
                return 0
            else: 
                while cap.isOpened():
                    _,img = cap.read()
                    frame,cnt_head = self.inference_img(img)
                    cv2.imshow('result',frame)
                    q=cv2.waitKey(10) & 0xFF
                    if q == 27 or q ==ord('q'):
                        break
            cap.release()
            cv2.destroyAllWindows()
        else:
            print('please input the right img-path')

if __name__ == '__main__':
    args = parms()
    detector = ModelTest(args)
    imgpath = args.file_in
    detector(imgpath)
    # evalu_img(args)