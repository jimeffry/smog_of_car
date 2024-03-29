#-*- coding:utf-8 -*-
import os
import sys
import cv2
import time
import torch
import argparse
import collections
import logging
from matplotlib import pyplot as plt
# import seaborn as sns;sns.set()
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfg
sys.path.append(os.path.join(os.path.dirname(__file__),'../prepare_data'))
from factory import dataset_factory, detection_collate
sys.path.append(os.path.join(os.path.dirname(__file__),'../resnest'))
from resnest import resnest101
sys.path.append(os.path.join(os.path.dirname(__file__),'../networks'))
from bisenetv2 import BiSeNetV2
from resnet_aspp import resnet_aspp50
from resnet_tr import resnet50
from mobilenet import mobilenet_v2
from shufflenet import shufflenet_v2_x1_0
from resnet_cbam import resnet18_cbam,resnet34_cbam,resnet50_cbam
from cspnet import cspresnext50
from resnet_plus import ig_resnext101_32x8d,ecaresnet101d,resnet18
from mlp_mixer import mixer_l16_224_in21k
sys.path.append(os.path.join(os.path.dirname(__file__),'../losses'))
from multiloss import MultiLoss,Focal_loss,LabelSmoothing,CrossEntropyLossOneHot


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
def params():
    parser = argparse.ArgumentParser(
        description='S3FD face Detector Training With Pytorch')
    train_set = parser.add_mutually_exclusive_group()
    parser.add_argument('--dataset',default='ShangHai',help='Train target')
    parser.add_argument('--start_epoch',default=0,type=int, help='training nums begin')
    parser.add_argument('--batch_size',default=2, type=int,help='Batch size for training')
    parser.add_argument('--resume',default=None, type=str,help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--num_workers',default=4, type=int,help='Number of workers used in dataloading')
    parser.add_argument('--cuda',default=True, type=str2bool,help='Use CUDA to train model')
    parser.add_argument('--lr', '--learning-rate',default=1e-3, type=float,help='initial learning rate')
    parser.add_argument('--cuda_num',default='0', type=str,help='Use CUDA to train model')
    parser.add_argument('--momentum',default=0.9, type=float,help='Momentum value for optim')
    parser.add_argument('--weight_decay',default=1e-4, type=float,help='Weight decay for SGD')
    parser.add_argument('--gamma',default=0.1, type=float,help='Gamma update for SGD')
    parser.add_argument('--multigpu',default=False, type=str2bool,help='Use mutil Gpu training')
    parser.add_argument('--save_folder',default='weights/',help='Directory for saving checkpoint models')
    parser.add_argument('--log_dir',default='../logs',help='Directory for saving logs')
    return parser.parse_args()


def renamedict(pretrained_state_dict):
    fsd = collections.OrderedDict()
    # 10 convlution *(weight, bias) = 20 parameters
    res_iter = pretrained_state_dict.items()
    for i in range(len(res_iter)):
        temp_key = list(res_iter)[i][0]
        # print(temp_key)
        if 'head' in temp_key:
            continue
        fsd[temp_key] = list(res_iter)[i][1]
    return fsd

def train_net(args):
    # if torch.cuda.is_available():
    #     if args.cuda:
    #         torch.set_default_tensor_type('torch.cuda.FloatTensor')
    #     if not args.cuda:
    #         print("WARNING: It looks like you have a CUDA device, but aren't " +
    #             "using CUDA.\nRun with --cuda for optimal training speed.")
    #         torch.set_default_tensor_type('torch.FloatTensor')
    # else:
    #     torch.set_default_tensor_type('torch.FloatTensor')
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    #*******load data
    train_dataset, val_dataset = dataset_factory(args.dataset)
    train_loader = data.DataLoader(train_dataset, args.batch_size,
                                num_workers=args.num_workers,
                                shuffle=True,
                                collate_fn=detection_collate,
                                pin_memory=True)
    val_batchsize = 8 #args.batch_size
    val_loader = data.DataLoader(val_dataset, val_batchsize,
                                num_workers=args.num_workers,
                                shuffle=False,
                                collate_fn=detection_collate,
                                pin_memory=True)
    #load net
    #print(args.cuda_num)
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_num
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    #net = resnet50(pretrained=True,num_classes=7).to(device)
    #net = resnet_aspp50(pretrained=True,num_classes=2).to(device)
    #net = resnest50(pretrained=False,num_classes=2).to(device)
    # net = mobilenet_v2(pretrained=True,num_classes=5).to(device)
    # net = shufflenet_v2_x1_0(pretrained=True,num_classes=2).to(device)
    #net = resnet50_cbam(pretrained=True,num_classes=5).to(device)
    #net = cspresnext50(pretrained=False,num_classes=2).to(device)
    #net = ig_resnext101_32x8d(pretrained=False,num_classes=2).to(device)
    # # net = ecaresnet101d(pretrained=False,num_classes=2).to(device)
    #net = resnest101(pretrained=False,num_classes=2).to(device)
    #net = BiSeNetV2(n_classes=2).to(device)
    #net = mixer_l16_224_in21k(pretrained=False,num_classes=2).to(device)
    net = resnet18(pretrained=True,num_classes=2).to(device)
    #print(">>",net)
    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        state_dict = torch.load(args.resume,map_location=device)
        #state_dict = renamedict(state_dict)
        net.load_state_dict(state_dict,strict=False)
        
    if args.multigpu:
        #net = torch.nn.DataParallel(net)
        #cudnn.benckmark = True
        if torch.cuda.device_count() > 1:
         #   print("******************* multi gpu run")
            torch.distributed.init_process_group(backend="nccl")
            net = torch.nn.parallel.DistributedDataParallel(net)
    #optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer = optim.Adam(net.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    criterion = LabelSmoothing(0.05) #Focal_loss() #LabelSmoothing(0.05) #MultiLoss()
    #criterion = CrossEntropyLossOneHot()
    print('Using the specified args:')
    print(args)
    return net,optimizer,criterion,train_loader,val_loader

def createlogger(lpath):
    if not os.path.exists(lpath):
        os.makedirs(lpath)
    logger = logging.getLogger()
    logname= time.strftime('%F-%T',time.localtime()).replace(':','-')+'.log'
    logpath = os.path.join(lpath,logname)
    hdlr = logging.FileHandler(logpath)
    logger.addHandler(hdlr)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    return logger

def main():
    args = params()
    logger = createlogger(args.log_dir)
    net,optimizer,criterion,train_loader,val_loader = train_net(args)
    step_index = 0
    start_epoch = args.start_epoch
    iteration = 0
    # net.train()
    tmp_diff = 0
    # rgb_mean = np.array([123.,117.,104.])[np.newaxis, np.newaxis,:].astype('float32')
    # rgb_std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis,:].astype('float32')
    rgb_mean = np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis,:].astype('float32')
    rgb_std = np.array([0.225, 0.225, 0.225])[np.newaxis, np.newaxis,:].astype('float32')
    loss_hist = collections.deque(maxlen=200)
    #set lr
    mode = 'cosineAnnWarm'
    if mode=='cosineAnn':
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    elif mode=='cosineAnnWarm':
        scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=2,T_mult=2)
    total_iter = len(train_loader)
    cur_lr_list = []
    print(total_iter)
    for epoch in range(start_epoch, cfg.EPOCHES):
        #losses = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
            save_fg = 0
            net.train()
            if args.cuda:
                images = images.cuda() 
                targets = targets.cuda()
            '''
            targets = targets.numpy()
            images = images.numpy()
            for i in range(targets.shape[0]):
                print(np.shape(images[i]))
                tmp_img = np.transpose(images[i],(1,2,0))
                tmp_img = tmp_img *rgb_std
                tmp_img = tmp_img + rgb_mean
                tmp_img = tmp_img * 255
                tmp_img = np.array(tmp_img,dtype=np.uint8)
                tmp_img = cv2.cvtColor(tmp_img,cv2.COLOR_RGB2BGR)
                h,w = tmp_img.shape[:2]
                gt = targets[i]
                print('gt label:',gt)
                cv2.imshow('src',tmp_img)
                plt.show()
                cv2.waitKey(0)
            '''
            # if iteration in cfg.LR_STEPS:
            #     step_index += 1
            #     adjust_learning_rate(args.lr,optimizer, args.gamma, step_index)
            # t0 = time.time()
            out = net(images)
            # backprop
            # optimizer.zero_grad()
            loss = criterion(out, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step(epoch + batch_idx / total_iter)
            cur_lr = optimizer.param_groups[-1]['lr']
            #cur_lr_list.append(cur_lr)
            # t1 = time.time()
            #print(targets.data.cpu().numpy()[:5])
            loss_hist.append(float(loss.item()))
            if iteration % 50 == 0 :
                #print('tl',loss.data,tloss)
                logger.info('epoch:{} || iter:{} || tloss:{:.6f},lossconf:{:.6f} || lr:{:.6f}'.format(epoch,iteration,np.mean(loss_hist),loss.item(),cur_lr))
                # val(args,net,val_loader,logger)
            if iteration > 0 and iteration % total_iter == 0:
                # sfile = 'csr_' + args.dataset + '_' + repr(iteration) + '.pth'
                sfile = 'broken_'+args.dataset+'_best_ecaresnet101d.pth'
                tmp_val = val(args,net,val_loader,logger)
                if tmp_val > tmp_diff:
                    save_fg = 1
                    tmp_diff = tmp_val
                logger.info('test acc:%.4f,  the best acc is: %.4f ' % (tmp_val,tmp_diff))
                if save_fg :
                    logger.info('Saving state, iter: %d' % iteration)
                    if args.multigpu:
                        torch.save(net.module.state_dict(),os.path.join(args.save_folder, sfile))
                    else:
                        torch.save(net.state_dict(),os.path.join(args.save_folder, sfile))
            iteration += 1
        if iteration == cfg.MAX_STEPS:
            break
        if epoch <= 50:
            optimizer.param_groups[-1]['lr'] = cfg.LR_Values[0]
        elif epoch > 50 and epoch <101:
            optimizer.param_groups[-1]['lr'] = cfg.LR_Values[1]
        elif epoch <151:
            optimizer.param_groups[-1]['lr'] = cfg.LR_Values[2]
        else:
            optimizer.param_groups[-1]['lr'] = cfg.LR_Values[3]
        #scheduler.step()
        
        #print(epoch)
    #plt.figure()
    #x_list = list(range(len(cur_lr_list)))
    #plt.plot(x_list, cur_lr_list)
    #plt.savefig('lr_test.png',format='png')
    #plt.show()

def val(args,net,val_loader,logger):
    net.eval()
    with torch.no_grad():
        t1 = time.time()
        eq_sum = 0.0
        for batch_idx, (images, targets) in enumerate(val_loader):
            targets = targets.long()
            if args.cuda:
                images = images.cuda()
                targets = targets.cuda()
            out = net(images).detach()
            out = F.softmax(out,dim=1)
            pred = torch.argmax(out,dim=1)
            pos_num = pred.eq(targets)
            tmp = pos_num.sum()
            eq_sum += tmp.item()
            # print(eq_sum)
        t2 = time.time()
        total_num = 8 *(batch_idx+1)
        print('Timer: %.4f' % (t2 - t1),'eq:',eq_sum,'total:',total_num)
        #logger.info('test acc:%.4f' % (eq_sum/total_num))
    return eq_sum/total_num



def adjust_learning_rate(init_lr,optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = init_lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
       