# tf-keras-yolo 
The tf-keras-yolo are used for training a YOLOv3 model on custom dasets by deploying keras and tensorflow. It is tested using tensorflow 1.15.3 and keras 2.2.4 (a higher eversion keras may need some modifications according to https://github.com/qqwweee/keras-yolo3/issues/544)

These scripts are adapdated from the fantastic work made by https://github.com/qqwweee/keras-yolo3 and https://github.com/AlexeyAB/darknet. See also https://github.com/experiencor/keras-yolo3 and https://github.com/wizyoung/YOLOv3_TensorFlow

# training
```
!python yoloTrain.py "--epoch" 100 "--classes_file" "image_data/train/yolov3_object_classes.txt" "--annotation_file" "image_data/train/yolov3_annotations.txt"
```

# prediction
```
!python yoloPredict.py "--input_path" "image_data/test" "--classes" "image_data/train/yolov3_object_classes.txt" "--yolo_model" "image_data/modelWeights/Logging/trained_weights_final.h5" "--output" "image_data/objectDetectionResults" "--box_file" "image_data/objectDetectionResults/detection.csv"
```
