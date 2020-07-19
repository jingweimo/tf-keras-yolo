# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 18:02:21 2020
Training YOLOv3 model based on pretrained weights 
                      and fine-tuning using custom data for object deteciton

#Environment requirements:
    tensorflow: 1.15
    keras: 2.2.4 (higher version needs be slightly problematic)
    python: <3.8
    cuda: 10.0 (higher version will be problematic)
    
#Usage: 
#Spyder
runfile("yoloTrain.py") #Spyder

#Colab (recommended due to the power of GCP)
!python yoloTrain.py "--epoch" 100 #Colab

@author: YL
"""

# environment setup
import os
import sys
import argparse

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import keras
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import (TensorBoard, 
                             ModelCheckpoint, 
                             ReduceLROnPlateau, 
                             EarlyStopping)

from yolo3.model import (preprocess_true_boxes, 
                         yolo_body, 
                         tiny_yolo_body, 
                         yolo_loss)
from yolo3.utils import get_random_data

from time import time
import pickle

#...............................Utility functions .............................
def get_parent_dir(n=0):
    """ return the n-th parent directory relative to the current working directory"""
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path
        
def get_classes(classes_path):
    """ load the object classes """
    with open(classes_path) as fID:
        class_names = fID.readlines()
    class_names = [item.strip() for item in class_names]
    return class_names

def get_anchors(anchors_path):
    """ load the anchors """
    with open(anchors_path) as fID:
        anchors = fID.readline()
    anchors = [float(x) for x in anchors.split(",")]
    #reshaped to 2-column array
    return np.array(anchors).reshape(-1, 2)

def create_model(input_shape, anchors, num_classes, load_pretrained=True,
                 freeze_body=2, weights_path="model_data/yolo.5"):
    """
    Create a YOLOv3 model
    
    Parameters
    ----------
    input_shape : image demension, e.g, (416, 416)
    anchors: 9 pairs of yolo anchors 
    num_classes : number of classes
    load_pretrained : TYPE, optional, The default is True.
    freeze_body : TYPE, optional, The default is 2 (182 layers frozen) (if set to 1, 185 layers forzen)
    weights_path : TYPE, optiona, The default is "model_data/yolo.5".

    Returns
    -------
    YOLOv3 model.
    """
    #start a new session
    K.clear_session()
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], 
                           w//{0:32, 1:16, 2:8}[l], 
                           num_anchors//3, 
                           num_classes+5
                           )
                    ) for l in range(3)
              ]

    #yolo_body
    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('\n Create YOLOv3 model with {} anchors and {} classes \n'.format(num_anchors, num_classes))
    
    #load pretrained model weights
    # https://keras.io/api/models/model_saving_apis/#load_weights-method
    """
    load_weigths: 
    With by_name set to True, weights are loaded into layers only if they share the same name. 
    This is useful for fine-tuning or transfer-learning models where some of the layers have changed
    The weights for the layers for customized data remain to be obtained through training
    """
    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('\n Load pretrained weights {}.\n'.format(weights_path))
        print("\n Before freezing, trainabled weights:{}\n".format(len(model_body.trainable_weights)))
        
        model_layer_num = len(model_body.layers)
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, model_layer_num-3)[freeze_body-1]
            for i in range(num): 
                model_body.layers[i].trainable = False
            print('\n Freeze the first {} layers of total {} layers.\n'.format(num, model_layer_num))
        print("\n After freezing, trainabled weights:{} \n".format(len(model_body.trainable_weights)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 
                                   'num_classes': num_classes, 
                                   'ignore_thresh': 0.5})([*model_body.output, *y_true])
    
    model = Model([model_body.input, *y_true], model_loss)

    return model
    

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, 
                      freeze_body=2,
                      weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], 
                           w//{0:32, 1:16}[l], 
                           num_anchors//2, 
                           num_classes+5)
                    ) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('\n Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    print("\n Before freezing, trainabled weights:{}".format(len(model_body.trainable_weights)))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('\n Load pretrained weights {}.\n'.format(weights_path))
        
        model_layer_num = len(model_body.layers)
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, model_layer_num-2)[freeze_body-1]
            for i in range(num): 
                model_body.layers[i].trainable = False
            print('\n Freeze the first {} layers of total {} layers.\n'.format(num, model_layer_num))
        print("\ After freezing, trainabled weights:{}\n".format(len(model_body.trainable_weights)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 
                                   'num_classes': num_classes, 
                                   'ignore_thresh': 0.7})([*model_body.output, *y_true])
    
    model = Model([model_body.input, *y_true], model_loss)

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)
        
def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)    

#...........................Pretrained Weights Conversion......................
#assuming yolo3.weights is downloaded to the current working directory
# !python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5 '-p'




#.................................Training.....................................
#default settings for the folder and path configration 
cdir = get_parent_dir(0)
print("Current directory: {}".format(cdir))
image_data_folder = os.path.join(cdir, "image_data") 
image_train_folder = os.path.join(image_data_folder, "train")
object_classes_path = os.path.join(image_train_folder, "yolov3_object_classes.txt")
objct_annotations_path = os.path.join(image_train_folder, "yolov3_annotations.txt")
#objct_annotations_path = os.path.join(image_train_folder, "yolov3_annotations_update.txt")
   
model_folder = os.path.join(image_data_folder, "modelWeights")

pretrained_weights_path = os.path.join(cdir,"model_data", "yolo.h5")
anchors_path = os.path.join(cdir,"model_data","yolo_anchors.txt")

#checkpoint folder
log_folder = os.path.join(model_folder,"Logging")
# if not os.path.exists(log_folder):
#     os.mkdir(log_folder)
if not os.path.exists(log_folder):
  os.makedirs(log_folder)

FLAGS = None
if __name__ =="__main__":
    #argument parsing
    #reseting
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    
    parser.add_argument(
        "--annotation_file",
        type=str,
        default=objct_annotations_path,
        help="Path to annotation file for Yolo. Default is " + objct_annotations_path,
    )
    parser.add_argument(
        "--classes_file",
        type=str,
        default=object_classes_path,
        help="Path to YOLO classnames. Default is " + object_classes_path,
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default=log_folder,
        help="Folder to save training logs and trained weights to. Default is " + log_folder,
    )

    parser.add_argument(
        "--anchors_path",
        type=str,
        default=anchors_path,
        help="Path to YOLO anchors. Default is " + anchors_path,
    )

    parser.add_argument(
        "--weights_path",
        type=str,
        default=pretrained_weights_path,
        help="Path to pre-trained YOLO weights. Default is " + pretrained_weights_path,
    )
    
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Percentage of training set to be used for validation. Default is 10%.",
    )
    
    parser.add_argument(
        "--is_tiny",
        default=False,
        action="store_true",
        help="Use the tiny Yolo version for better performance and less accuracy. Default is False.",
    )
    
    parser.add_argument(
        "--random_seed",
        type=float,
        default=None,
        help="Random seed value to make script deterministic. Default is 'None', i.e. non-deterministic.",
    )
    
    parser.add_argument(
        "--epochs",
        type=float,
        default=51,
        help="Number of epochs for training last layers and number of epochs for fine-tuning layers. Default is 51.",
    )
    
    #argument flags
    FLAGS = parser.parse_args()
    print("Class file: {}".format(FLAGS.classes_file))

    np.random.seed(FLAGS.random_seed)

    log_dir = FLAGS.log_dir

    class_names = get_classes(FLAGS.classes_file)
    num_classes = len(class_names)
    print("\n Found {} class: {}\n".format(num_classes,class_names))

    anchors = get_anchors(FLAGS.anchors_path)
    weights_path = FLAGS.weights_path

    input_shape = (416, 416)  # multiple of 32, height, width
    epoch1, epoch2 = FLAGS.epochs, FLAGS.epochs #adjustable
 
    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes, 
                                  freeze_body=2, 
                                  weights_path=weights_path)
    else:
        #create YOLOv3 model
        model = create_model(input_shape, anchors, num_classes,
                             freeze_body=2, 
                             weights_path=weights_path) # make sure you know what you freeze

    log_dir_time = os.path.join(log_dir, "{}".format(int(time()))) #for saving model history
    #logging = TensorBoard(log_dir=log_dir)
    logging = TensorBoard(log_dir=log_dir_time)
    
    #callbacks
    #1. checkpoint (https://machinelearningmastery.com/check-point-deep-learning-models-keras/)
    # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
    # https://keras.io/api/callbacks/model_checkpoint/
    # https://stackoverflow.com/questions/59069058/save-model-every-10-epochs-tensorflow-keras-v2
    filepath = os.path.join(log_dir,'yolov3Weights.ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5')
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_loss', 
                                 save_weights_only=True, 
                                 save_best_only=True, 
                                 period=5)
    #2. learing rate
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)    
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    #custom callback
    # https://keras.io/guides/writing_your_own_callbacks/
    # https://github.com/keras-team/keras/issues/7874
    # https://stackoverflow.com/questions/49785536/get-learning-rate-of-keras-model
    class lrepPrintout(keras.callbacks.Callback):
        # called at the end of an epoch during training
        def on_epoch_end(self, epoch, logs=None):
          lr = float(K.eval(self.model.optimizer.lr))
          print("\n Epoch#{:3d}: learning rate={:2.4f}\n".format(epoch, lr))

    val_split = FLAGS.val_split # set the size of the validation set
    with open(FLAGS.annotation_file) as f:
        lines = f.readlines()

    
    #np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a decent model.
    if True:
        model.compile(optimizer=Adam(lr=1e-3), 
                      loss={# use custom yolo_loss Lambda layer.
                            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 32 #Be aware of OOM
        print('\n Stage#1: Train on {} samples, val on {} samples, with batch size {}.\n'.format(num_train, num_val, batch_size))
        #start training
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train//batch_size),
                            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                            validation_steps=max(1, num_val//batch_size),
                            epochs=epoch1,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint, lrepPrintout()])
        
        #save model weights
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')
        
        #training and validation loss




    print("\n Unfreeze and continue training to fine-tune the model \n")
    # https://github.com/qqwweee/keras-yolo3/issues/191
    # https://github.com/qqwweee/keras-yolo3/issues/89
    # https://www.programmersought.com/article/6237642638/
    # https://www.cnblogs.com/greentomlee/p/9843258.html

    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), 
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        
        print('Unfreeze all of the layers.')

        #batch_size = 32 # note that more GPU memory is required after unfreezing the body
        batch_size = 8
        print('\n Stage#2: Train on {} samples, val on {} samples, with batch size {}.\n'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train//batch_size),
                            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                            validation_steps=max(1, num_val//batch_size),
                            epochs=epoch1+epoch2,
                            initial_epoch=epoch1,
                            callbacks=[logging, checkpoint, reduce_lr, early_stopping, lrepPrintout()])
        
        #saving the final weights
        model.save_weights(os.path.join(log_dir, 'trained_weights_final.h5'))

    print('\n YOLO Trained Weights Saved to {} \n'.format(os.path.join(log_dir,'trained_weights_final.h5')))
    print('\n YOLO Training is Done!!')
    
    #Training and validation loss
    
    
    #IoU metrics