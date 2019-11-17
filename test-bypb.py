#coding:utf-8
import tensorflow as tf
import numpy as np
import cv2
import pickle as pkl
import os
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model = 7
input_size = (600,600)

def GetFileList(dir,filelist):
    new_dir = dir
    if os.path.isfile(dir):
        filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            new_dir = os.path.join(dir,s)
            GetFileList(new_dir,filelist)
    return filelist

f = open('labels_map.txt','r')
labels = eval(f.readline())

image = cv2.imread('panda.jpg')
image = cv2.resize(image,input_size)
# image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
input_data = np.array(image,dtype=np.float32)[np.newaxis,:,:,:]
print('image',input_data.shape)

def inference():
    tf.reset_default_graph()  # 重置计算图
    output_graph_path = './EfficientNet-b%d.pb'%model
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        output_graph_def = tf.GraphDef()
        # 获得默认的图
        graph = tf.get_default_graph()
        with open(output_graph_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")

            inputs = tf.get_default_graph().get_tensor_by_name('IteratorGetNext:0')
            output = tf.get_default_graph().get_tensor_by_name('efficientnet-b%d/model/head/dense/BiasAdd:0'%model)
            output = tf.nn.softmax(output)
            output = tf.squeeze(output)

            start = time.time()
            output = sess.run(output,feed_dict={inputs:input_data})
            output = np.ravel(output)
            loc = np.argsort(output)[::-1][:5]
            output = [output[pid] for pid in loc]
            for i in range(5):
                print('TOP_%d:'%i,'%2.2f'%(output[i]*100),'%  ',labels[str(loc[i])])

            time_used = time.time()-start
            print("%.3f s used\n"%time_used)

def middle_node():
    tf.reset_default_graph()  # 重置计算图
    output_graph_path = './EfficientNet-b%d.pb'%model
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        output_graph_def = tf.GraphDef()
        # 获得默认的图
        graph = tf.get_default_graph()
        with open(output_graph_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")

            inputs = tf.get_default_graph().get_tensor_by_name('IteratorGetNext:0')
            # output = tf.get_default_graph().get_tensor_by_name('efficientnet-b%d/model/blocks_0/se/conv2d_1/BiasAdd:0'%model)
            output = tf.get_default_graph().get_tensor_by_name('efficientnet-b7/model/blocks_38/tpu_batch_normalization_2/FusedBatchNorm:0')
            results = sess.run(output,feed_dict={inputs:input_data})
            print("results:")
            print("shape:",results.shape)
            print("min,max value:",np.min(results),np.max(results))
            print("the first 20 numbers:",np.ravel(results)[0:20])

if __name__=="__main__":
    inference()
    # middle_node()