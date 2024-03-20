import os
import sys
import cv2
import torch
import onnx
from onnx_tf.backend import prepare
# import tensorflow as tf
import numpy as np
from torch.autograd import Variable
sys.path.append(os.path.join(os.path.dirname(__file__),'../networks'))
from resnet_tr import resnet50,resnet18
from mobilenet import mobilenet_v2
from shufflenet import shufflenet_v2_x1_0
from cspnet import cspresnext50
sys.path.append(os.path.join(os.path.dirname(__file__),'../resnest'))
from resnest import resnest50

def rename_dict(state_dict):
    state_dict_new = dict()
    for key,value in list(state_dict.items()):
        state_dict_new[key[7:]] = value
    return state_dict_new

def tr2onnx(modelpath):
    # Load the trained model from file
    device = 'cpu'
    # net = resnet50(pretrained=False,num_classes=3).to(device)
    #net = resnest50(pretrained=False,num_classes=3).to(device)
    # net = shufflenet_v2_x1_0(pretrained=False,num_classes=6)
    # net = resnet18(pretrained=False,num_classes=2).to(device)
    # net = resnest50(pretrained=False,num_classes=3).to(device)
    # net = mobilenet_v2(pretrained=False,num_classes=6).to(device)
    net = cspresnext50(pretrained=False,num_classes=2).to(device)
    state_dict = torch.load(modelpath,map_location=device)
    # state_dict = rename_dict(state_dict)
    net.load_state_dict(state_dict)
    net.eval()
    # Export the trained model to ONNX
    dummy_input = Variable(torch.randn(1,3,448,448)) # 8 x 28 picture will be the input to the model
    export_onnx_file = '../models/broken_cspresnext50.onnx'
    torch.onnx.export(net,
                    dummy_input,
                    export_onnx_file,
                    opset_version=10,
                    do_constant_folding=True, # 是否执行常量折叠优化
                    input_names=["img_input"], # 输入名
                    output_names=["softmax_output"], # 输出名
                    dynamic_axes={"img_input":{0:"batch_size"}, # 批处理变量
                                    "softmax_output":{0:"batch_size"}}
    )

def onnxresave(modelpath):
    from onnx import shape_inference
    model = '../models/best.onnx'
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(modelpath)), model)
    print("*******over")

def onnx2tf(modelpath):
    # Load the ONNX file
    print("**********begin load")
    model = onnx.load(modelpath)
    print("************onnx over")
    # Import the ONNX model to Tensorflow
    tf_rep = prepare(model)
    print("**********tf load")
    # Input nodes to the model
    print('inputs:', tf_rep.inputs)
    # Output nodes from the model
    print('outputs:', tf_rep.outputs)
    # All nodes in the model
    print('tensor_dict:')
    # print(tf_rep.tensor_dict)
    # 运行tensorflow模型
    print('Image 1:')
    # img = cv2.imread('/data/detect/breathmask/fg1/1_0.jpg')
    # img = np.transpose(img,(2,0,1))
    # output = tf_rep.run(np.asarray(img, dtype=np.float32)[np.newaxis,:,:, :])
    # print('The digit is classified as ', np.argmax(output))
    tf_rep.export_graph('..\models\mobilev2.pb')

def convert_pbtxt_to_pb(filename):
    from google.protobuf import text_format
    """Returns a `tf.GraphDef` proto representing the data in the given pbtxt file.
    Args:
    filename: The name of a file containing a GraphDef pbtxt (text-formatted
      `tf.GraphDef` protocol buffer data).
    """
    with tf.gfile.FastGFile(filename, 'r') as f:
        graph_def = tf.GraphDef()
        img_input = tf.placeholder(dtype=tf.float32,shape=[None,3,112,112], name='img_input') 
        file_content = f.read()

        # Merges the human-readable string in `file_content` into `graph_def`.
        text_format.Merge(file_content, graph_def)
        tf.train.write_graph( graph_def , './' , 'breathtest.pb' , as_text = False )

def pbtxt_to_graphdef(filename):
    import tensorflow as tf
    from google.protobuf import text_format
    with open(filename, 'r') as f:
        graph_def = tf.GraphDef()
        file_content = f.read()
        text_format.Merge(file_content, graph_def)
        tf.import_graph_def(graph_def, name='')
        tf.train.write_graph(graph_def, './', '../models/mobile_bs.pb', as_text=False)


def graphdef_to_pbtxt(filename):
    '''
    find 'input' replace batch_dim with '-1'
    '''
    import tensorflow as tf
    from tensorflow.python.platform import gfile
    with gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        tf.train.write_graph(graph_def, './', '../models/mobile_tmp.pbtxt', as_text=True)


if __name__=='__main__':
    # modelpath = '/data/models/breathmask/bm_mafa_best.pth'
    # tr2onnx(modelpath)
    # modelpath = '../models/breathmask1.onnx'
    # onnx2tf(modelpath)
    # tr2onnx('D:\models\mobile_allian_best_res50_allv2.pth')
    # onnx2tf('..\models\mobile.onnx')
    # graphdef_to_pbtxt('..\models\mobile_tf14.pb')
    #pbtxt_to_graphdef('../models/mobile_tmp.pbtxt')
    # tr2onnx("D:\models\\broken_allian_best_cspresnext50.pth")
    onnxresave("../models/broken_cspresnext50.onnx")