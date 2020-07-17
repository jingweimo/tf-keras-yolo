# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 08:24:10 2020

Apply a custom-trained YOLOv3 model for object detection:
    runfile('yoloPredict.py')

The detection results are saved to:
    detectionResults, including the raw images with bounding boxes and confidence scores
                      and a csv file storing the image file path and detection results

For video, the output files are videos with bounding boxes and confidence scores

See also: yolo.py
@author: Daisy
"""

import os
import sys

def getParentDir(n=0):
    """
    Parameters
    ----------
    n : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    The n-th parent directory of the current working directory.

    """
    cdir = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        cdir = os.path.dirname(cdir)
        
    #print('The {}-th directory: {}\n'.format(n,cdir))
    return cdir

# train/, test/ and valid/, with annotations and classes

def getFileList(dirName, endings=[".jpg", ".jpeg", ".png", ".mp4"]):
    # create a list of file and sub directories names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    
    # Make sure all file endings start with a '.'
    for i, ending in enumerate(endings):
        if ending[0] != ".":
            endings[i] = "." + ending
            
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getFileList(fullPath, endings)
        else:
            for ending in endings:
                if entry.endswith(ending):
                    allFiles.append(fullPath)
    return allFiles

#..............................................................................
import argparse
import numpy as np
import pandas as pd

from yolo import YOLO, detect_video 
from PIL import Image
import matplotlib.pyplot as plt
from timeit import default_timer as timer

def yolo_detect_object(yolo, img_path, save_img, save_img_path="./", postfix=""):
    """    
    Apply YOLOv3 to detect objects in an image 
    Parameters
    ----------
    yolo : keras-yolo3 YOLO instance initialized with custom-trained weights.
    img_path : path to the image to be detected
    save_img : bool to save the image or not
    save_img_path : path to the directory where to save the image. The default is "./".
    postfix : TYPE, optional, The default is "".

    Returns
    -------
    Predictions and updated image.
    
    See also: yolo.detect_image()
    """
    try:
        #imgPath
        image = Image.open(img_path)
        if image.mode != "RGB":
            image = image.covert("RGB")
        image_array = np.array(image)
    except:
        print("File Open Error! Try Again!")
        return None, None

    #objet detection
    plt.imshow(image)
    plt.show()
    prediction, newImage = yolo.detect_image(image)
    
    #save
    img_out = postfix.join(os.path.splitext(os.path.basename(img_path)))
    if save_img:
      newImage.save(os.path.join(save_img_path,img_out))
    
    return prediction, image_array

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

#set up default folder names
#img_folder = os.path.join(getParentDir(0),"Images")
img_folder = 'test/'
#obj_classes = os.path.join(img_folder,'object_classes.txt')
obj_classes = "test/_classes.txt"
#img_test_folder = os.path.join(img_folder,'test')
img_test_folder = img_folder
img_detection_folder = os.path.join(img_folder,'detectionResults')
img_detection_file = os.path.join(img_detection_folder,'detectionResutls.csv')

#model_folder = os.path.join(img_folder,"modelWeights")
#model_weights = os.path.join(model_folder,"trained_weights_final.h5")
model_weights = "trained_weights_final.h5"

anchors_path = os.path.join(getParentDir(0),"model_data","yolo_anchors.txt")

print("Test Image folder: ", img_test_folder)

#..............................................................................
FLAGS = None
if __name__=='__main__':
    #argument parsing (https://docs.python.org/2/library/argparse.html#argument-default)
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    
    parser.add_argument(
        "--input_path",
        type=str,
        default=img_test_folder,
        help="Path to image/video directory. All subdirectories will be included. Default is "
        + img_test_folder,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=img_detection_folder,
        help="Output path for detection results. Default is "
        + img_detection_folder,
    )

    parser.add_argument(
        "--no_save_img",
        default=False,
        action="store_true",
        help="Only save bounding box coordinates but do not save output images with annotated boxes. Default is False.",
    )

    parser.add_argument(
        "--file_types",
        "--names-list",
        nargs="*",
        default=[],
        help="Specify list of file types to include. Default is --file_types .jpg .jpeg .png .mp4",
    )

    parser.add_argument(
        "--yolo_model",
        type=str,
        dest="model_path",
        default=model_weights,
        help="Path to pre-trained weight files. Default is " + model_weights,
    )

    parser.add_argument(
        "--anchors",
        type=str,
        dest="anchors_path",
        default=anchors_path,
        help="Path to YOLO anchors. Default is " + anchors_path,
    )

    parser.add_argument(
        "--classes",
        type=str,
        dest="classes_path",
        default=obj_classes,
        help="Path to YOLO class specifications. Default is " + obj_classes,
    )

    parser.add_argument(
        "--gpu_num", type=int, default=1, help="Number of GPU to use. Default is 1"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        dest="score",
        default=0.25,
        help="Threshold for YOLO object confidence score to show predictions. Default is 0.25.",
    )

    parser.add_argument(
        "--box_file",
        type=str,
        dest="box",
        default=img_detection_file,
        help="File to save bounding box results to. Default is "
        + img_detection_file,
    )

    parser.add_argument(
        "--postfix",
        type=str,
        dest="postfix",
        default="_object",
        help='Specify the postfix for images with bounding boxes. Default is "_object"',
    )
    
    FLAGS = parser.parse_args()
    save_img = not FLAGS.no_save_img
    file_types = FLAGS.file_types
    #print(file_types) #[]
    
    input_path = FLAGS.input_path
    if file_types:
        input_paths = getFileList(input_path, endings=file_types)
    else:
        input_paths = getFileList(input_path)   
    
    #Split files into images and videos
    img_exts = (".jpg", ".jpeg", ".png")
    vid_exts = (".mp4", ".mpeg", ".mpg", ".avi")
    
    input_img_paths = []
    input_vid_paths = []
    for item in input_paths:
        if item.endswith(img_exts):
            input_img_paths.append(item)
        elif item.endswith(vid_exts):
            input_vid_paths.append(item)
    
    #output path for saving
    output_path = FLAGS.output
    if not os.path.exists(output_path):
        os.makedirs(output_path)
     
    #class labels
    class_file = open(FLAGS.classes_path,"r")
    input_labels = [line.rstrip("\n") for line in class_file.readlines()]
    class_file.close()
    print("Found {} object labels: {}".format(len(input_labels), input_labels))  
    
    #..........................................................................
    #YOLO detector
    yolo_model_path = FLAGS.model_path
    #print("Yolo model weights to use:\n", yolo_model_path)
    myYOLO = YOLO(**{"model_path":yolo_model_path,
                     "anchors_path":FLAGS.anchors_path,
                     "classes_path":FLAGS.classes_path,
                     "score":FLAGS.score,
                     "gpu_num":FLAGS.gpu_num,
                     "model_image_size":(416,416) #depending on the actual size
        })
    
    #Dataframe for prediction output
    outDF = pd.DataFrame(
        columns = [
            "image",
            "image_path",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "label",
            "confidence",
            "xsize",
            "ysize",
            ]
        )
    
    #Image object prediction
    if(input_img_paths):
        print('Found {} input images {}'.format(
            len(input_img_paths),[os.path.basename(f) for f in input_img_paths[:5]]))
    
        start = timer()
        text_out = ""
        for i, img_path in enumerate(input_img_paths):
            print("The image to predict: ", img_path)
            prediction, image = yolo_detect_object(myYOLO, 
                                                   img_path,
                                                   save_img=save_img,
                                                   save_img_path=output_path,
                                                   postfix=FLAGS.postfix)
            
            ysize, xsize, _ = np.array(image).shape
            for single_prediction in prediction:
                outDF = outDF.append(pd.DataFrame(
                    [[os.path.basename(img_path.rstrip("\n")), img_path.rstrip("\n")]
                     + single_prediction
                     + [xsize, ysize]],
                    columns = [
                        "image",
                        "image_path",
                        "xmin",
                        "ymin",
                        "xmax",
                        "ymax",
                        "label",
                        "confidence",
                        "xsize",
                        "ysize",
                        ]))
                
        end = timer()
        print("Processed {} images in {:.1f}sec - {:.1f}FPS".format(
            len(input_img_paths), end - start, 
            len(input_img_paths) / (end - start)))
            
        outDF.to_csv(FLAGS.box, index=False)
        
    myYOLO.close_session()
    
    
    
    
    
