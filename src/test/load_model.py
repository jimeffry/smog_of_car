###############################################
#created by :  lixiaoyu80
#Time:  2021/1/12 19:09
#project: mobilephone recognition
#rversion: 0.1
#tool:   python 3.6
#modified:
#description  
####################################################
from tensorflow.python.platform import gfile
import tensorflow as tf
# from keras.applications import MobileNetV2
# from keras import layers
# from keras.models import Model,load_model
import sys
import os
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable

import cv2
import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__),'..\configs'))
from config import cfg
sys.path.append(os.path.join(os.path.dirname(__file__),'../networks'))
from resnet_tr import resnet50,resnet18
from mobilenet import mobilenet_v2
from shufflenet import shufflenet_v2_x1_0
from resnet_cbam import resnet50_cbam
from resnet_plus import ecaresnet101d
from cspnet import cspresnext50
from mlp_mixer import mixer_l16_224_in21k
import random
seed=0
 
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
 
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
 
# Remove randomness (may be slower on Tesla GPUs) 
# https://pytorch.org/docs/stable/notes/randomness.html
if seed == 0:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parms():
    parser = argparse.ArgumentParser(description='CSRnet demo')
    parser.add_argument('--save_dir', type=str, default='tmp/',
                        help='Directory for detect result')
    parser.add_argument('--modelpath', type=str,
                        default='weights/s3fd.pth', help='trained model')
    parser.add_argument('--threshold', default=0.65, type=float,
                        help='Final confidence threshold')
    parser.add_argument('--cuda_use', default=False, type=bool,
                        help='gpu run')
    parser.add_argument('--img_dir', type=str, default='tmp/',
                        help='Directory for images')
    parser.add_argument('--file_in', type=str, default='tmp.txt',
                        help='image namesf')
    return parser.parse_args()



class MobilePhoneNet(object):
    def __init__(self,args):
        self.loadmodel(args.modelpath)
        self.real_num = 0

    def loadmodel(self,modelpath):
        self.net = load_model(modelpath)

    def propress(self,imgs):
        # rgb_mean = np.array([123.,117.,104.])[np.newaxis, np.newaxis,:].astype('float32')
        rgb_mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis,:].astype('float32')
        rgb_std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis,:].astype('float32')
        img_out = []
        for img in imgs:
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            w,h = img.shape[:2]
            if w != cfg.InputSize_w or h != cfg.InputSize_h:
                img = cv2.resize(img,(cfg.InputSize_w,cfg.InputSize_h))
            img = img.astype('float32')
            img /= 255.0
            # img -= rgb_mean
            # img /= rgb_std
            # img = np.transpose(img,(2,0,1))
            img_out.append(img)
        return np.array(img_out)

    def inference(self,imglist):
        t1 = time.time()
        bt_img = self.propress(imglist)
        output = self.net.predict(bt_img)
        pred_cls = np.argmax(output,axis=1)
        # scores = [output[idx,pred_cls[idx]].data.cpu().numpy() for idx in range(output.size(0))]
        t2 = time.time()
        # print('consuming:',t2-t1)
        return output,pred_cls

class MobilePhoneNetTR(object):
    def __init__(self,args):
        if args.cuda_use and torch.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False
        if self.use_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
        self.loadmodel(args.modelpath)
        self.register()
        self.real_num = 0
        self.features_blobs = []

    def loadmodel(self,modelpath):
        if self.use_cuda:
            device = 'cuda'
        else:
            device = 'cpu'
        # self.net = shufflenet_v2_x1_0(pretrained=False,num_classes=6).to(device)
        # self.net = resnet50(pretrained=False,num_classes=3).to(device)
        # self.net = mobilenet_v2(pretrained=False,num_classes=6).to(device)
        # self.net = resnet50_cbam(pretrained=False,num_classes=3).to(device)
        # self.net = cspresnext50(pretrained=False,num_classes=2).to(device)
        # self.net = ecaresnet101d(pretrained=False,num_classes=2).to(device)
        # # self.net = mixer_l16_224_in21k(pretrained=False,num_classes=2).to(device)
        # state_dict = torch.load(modelpath,map_location=device)
        # self.net.load_state_dict(state_dict)
        self.net = torch.jit.load(modelpath).to(device)
        self.net.eval()
        # if self.use_cuda:
        #     cudnn.benckmark = True
        # torch.save(self.net.state_dict(),'rbcar_best.pth')
        # print("***********begin to resave")
        # example = torch.rand(1, 3, 448, 448)
        # example = Variable(example.to('cpu'))
        # traced_script_module = torch.jit.trace(self.net.to('cpu'), (example))
        # # # traced_script_module = torch.jit.script(self.net)
        # traced_script_module.save(modelpath[:-4]+".pt")
        # print("**************save over")
        # self.net.to(device)

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

    def propress(self,imgs):
        # rgb_mean = np.array([123.,117.,104.])[np.newaxis, np.newaxis,:].astype('float32')
        rgb_mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis,:].astype('float32')
        rgb_std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis,:].astype('float32')
        img_out = []
        '''
        imgs_split = []
        hhalf = int(imgh/2)
        whalf = int(imgw/2)
        if imgh>imgw:
            imgs_split.append(imgs[0][:hhalf,:,:])
            imgs_split.append(imgs[0][hhalf:,:,:])
        else:
            imgs_split.append(imgs[0][:,:whalf,:])
            imgs_split.append(imgs[0][:,whalf:,:])
        '''
        for img in imgs:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            w,h = img.shape[:2]
            if w != cfg.InputSize_w or h != cfg.InputSize_h:
                img = cv2.resize(img,(cfg.InputSize_w,cfg.InputSize_h))
                # img = self.resize_scale(img,(cfg.InputSize_w,cfg.InputSize_h))
            img = img.astype('float32')
            img /= 255.0
            img -= rgb_mean
            img /= rgb_std
            img = np.transpose(img,(2,0,1))
            img_out.append(img)
        return np.array(img_out)

    def hook_feature(self,module, input, output):
        self.features_blobs.append(output.data.cpu().numpy())
    def register(self):
        # self.net._modules.get(finalconv_name).register_forward_hook(self.hook_feature)
        params = list(self.net.parameters())
        # for tmp in params:
        #     print(tmp.cpu().data.numpy().shape)
        self.weight_softmax = np.squeeze(params[-2].data.cpu().numpy()) # resnet50
        # weight_softmax = np.squeeze(params[-2].data.cpu().numpy()) #ecaresnet101
        
    def getCam(self,img,class_idx):
        '''
        img: original image
        class_idx: list, be interest in
        return: hot image
        '''
        height,width,_ = img.shape
        feature_conv = self.features_blobs[0]
        bz, nc, h, w = feature_conv.shape
        # print("w_softmax and feature conv h w",self.weight_softmax.shape,feature_conv.shape)
        feature_conv = feature_conv[0]
        output_cam = []
        for idx in class_idx:
            cam = self.weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            # cam_img = cv2.resize(cam_img, (cfg.InputSize_w,cfg.InputSize_h))
            cam_img = cv2.resize(cam_img,(width,height))
            heatmap = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
            # print(heatmap.shape,img.shape)
            result = heatmap * 0.3 + img * 0.7
            # output_cam.append(result)
            output_cam.append([cam_img,result])
        return output_cam

    def inference(self,imglist):
        t1 = time.time()
        self.features_blobs = []
        imgs = self.propress(imglist)
        bt_img = torch.from_numpy(imgs)
        if self.use_cuda:
            bt_img = bt_img.cuda()
        # self.register('forward_features')
        # print(self.net)
        output,logits = self.net(bt_img)
        self.features_blobs.append(logits.data.cpu().numpy())
        #output = F.softmax(output,dim=-1)
        pred_cls = torch.argmax(output,dim=1)
        # scores = [output[idx,pred_cls[idx]].data.cpu().numpy() for idx in range(output.size(0))]
        t2 = time.time()
        # print('consuming:',t2-t1)
        # if pred_cls[0]==1 and pred_cls[1]==1:
        #     output[0] = output.max(axis=0)[0]
        # elif pred_cls[1] ==1 :
        #     output[0] = output[1]
        #     pred_cls[0] = pred_cls[1]
        return output.data.cpu().numpy(),pred_cls.data.cpu().numpy()

class MobilePhoneNetTF(object):
    def __init__(self,args):
        self.loadmodel(args.modelpath)
        
    def loadmodel(self,mpath):
        print("*******************begin to load")
        tf_config = tf.ConfigProto()
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        #tf_config.gpu_options = gpu_options
        tf_config.gpu_options.allow_growth=True  
        tf_config.log_device_placement=False
        self.sess = tf.Session(config=tf_config)
        # self.sess = tf.Session()
        modefile = gfile.FastGFile(mpath, 'rb')
        graph_def = tf.GraphDef()
        #graph_def = tf.compat.v1.GraphDef()
        print("****************** read over")
        graph_def.ParseFromString(modefile.read())
        print("******************loading")
        self.sess.graph.as_default()
        tf.import_graph_def(graph_def, name='') 
        # tf.train.write_graph(graph_def, './', 'breathtest.pbtxt', as_text=True)
        # print("************begin to print graph*******************")
        # op = self.sess.graph.get_operations()
        # for m in op:
        #     # if 'input' in m.name or 'output' in m.name or 'confidence' in m.name:
        #     print(m.name)#m.values())
        # print("********************end***************")
        self.input_image = self.sess.graph.get_tensor_by_name('img_input:0') #img_input
        self.conf_out = self.sess.graph.get_tensor_by_name('softmax_output:0') #softmax_output

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

    def propress(self,imgs):
        rgb_mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis,:].astype('float32')
        rgb_std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis,:].astype('float32')
        img_out = []
        for img in imgs:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            w,h = img.shape[:2]
            if w != cfg.InputSize_w or h != cfg.InputSize_h:
                # img = cv2.resize(img,(cfg.InputSize_w,cfg.InputSize_h))
                img = self.resize_scale(img,(cfg.InputSize_w,cfg.InputSize_h))
            img = img.astype('float32')
            img /= 255.0
            img -= rgb_mean
            img /= rgb_std
            # img = np.transpose(img,(2,0,1))
            img_out.append(img)
        return np.array(img_out)

        
    def inference(self,imglist):
        t1 = time.time()
        bt_img = self.propress(imglist)
        # output = []
        # for i in range(bt_img.shape[0]):
        #     tmp_output = self.sess.run([self.conf_out],feed_dict={self.input_image:np.expand_dims(bt_img[i],0)})
        #     output.append(tmp_output[0][0])
        output = self.sess.run([self.conf_out],feed_dict={self.input_image:bt_img})
        # t2 = time.time()
        # print("debug*********",np.shape(output))
        output = np.array(output[0])
        pred_cls = np.argmax(output,axis=1)
        t3 = time.time()
        # print('consuming:',t3-t1)
        # showimg = self.label_show(bboxes,imgorg)
        return output,pred_cls,bt_img

class PhoneAttributeTF(object):
    def __init__(self,args):
        self.loadmodel(args.modelpath)
        
    def loadmodel(self,mpath):
        print("*******************begin to load")
        tf_config = tf.ConfigProto()
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        #tf_config.gpu_options = gpu_options
        tf_config.gpu_options.allow_growth=True  
        tf_config.log_device_placement=False
        self.sess = tf.Session(config=tf_config)
        # self.sess = tf.Session()
        modefile = gfile.FastGFile(mpath, 'rb')
        graph_def = tf.GraphDef()
        #graph_def = tf.compat.v1.GraphDef()
        print("****************** read over")
        graph_def.ParseFromString(modefile.read())
        print("******************loading")
        self.sess.graph.as_default()
        tf.import_graph_def(graph_def, name='') 
        # tf.train.write_graph(graph_def, './', 'breathtest.pbtxt', as_text=True)
        # print("************begin to print graph*******************")
        # op = self.sess.graph.get_operations()
        # for m in op:
        #     # if 'input' in m.name or 'output' in m.name or 'confidence' in m.name:
        #     print(m.name)#m.values())
        # print("********************end***************")
        self.input_image = self.sess.graph.get_tensor_by_name('input_1:0') #img_input
        self.broken_out = self.sess.graph.get_tensor_by_name('dense_1/Softmax:0') #softmax_output
        self.light_out = self.sess.graph.get_tensor_by_name('dense_2/Softmax:0')
        self.shape_out = self.sess.graph.get_tensor_by_name('dense_3/Softmax:0')

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

    def propress(self,imgs):
        rgb_mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis,:].astype('float32')
        rgb_std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis,:].astype('float32')
        img_out = []
        for img in imgs:
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            w,h = img.shape[:2]
            if w != cfg.InputSize_w or h != cfg.InputSize_h:
                img = cv2.resize(img,(640,640))
                # img = self.resize_scale(img,(cfg.InputSize_w,cfg.InputSize_h))
            img = img.astype('float32')
            img /= 255.0
            # img -= rgb_mean
            # img /= rgb_std
            # img = np.transpose(img,(2,0,1))
            img_out.append(img)
        return np.array(img_out)

        
    def inference(self,imglist):
        t1 = time.time()
        bt_img = self.propress(imglist)
        # output = []
        # for i in range(bt_img.shape[0]):
        #     tmp_output = self.sess.run([self.conf_out],feed_dict={self.input_image:np.expand_dims(bt_img[i],0)})
        #     output.append(tmp_output[0][0])
        output = self.sess.run([self.broken_out,self.light_out,self.shape_out],feed_dict={self.input_image:bt_img})
        # t2 = time.time()
        # print("debug*********",np.shape(output))
        broken_output = np.array(output[0])
        light_out = np.array(output[1])
        shape_out = np.array(output[2])
        broken_cid = np.argmax(broken_output,axis=1)
        light_cid = np.argmax(light_out,axis=1)
        shape_cid = np.argmax(shape_out,axis=1)
        t3 = time.time()
        out_cls = np.array([broken_cid,light_cid,shape_cid]).transpose()
        out_score = np.concatenate((broken_output[:,broken_cid],light_out[:,light_cid],shape_out[:,shape_cid]),axis=1)
        print('consuming:',t3-t1)
        # showimg = self.label_show(bboxes,imgorg)
        return out_score,out_cls #bt_img

if __name__ == '__main__':
    args = parms()
    net = MobilePhoneNetTR(args)