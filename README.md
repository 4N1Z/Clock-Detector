
# Clock Detector Model using YOLO v5
This is a clock detector model trained using the YOLO v5 algorithm. The model is able to detect clocks in images and provide bounding boxes around the objects.

### Dataset
The custom dataset was created using Roboflow platform, by giving annotations to the clock images. The dataset consists of 1000 images of clocks of various types and styles. The images were split into training and validation sets with an 80/20 split ratio.

### Model Training
The model was trained using the YOLOv5 algorithm with the PyTorch framework. The training process was carried out on Google Colab, with a GPU runtime to accelerate training. The training process involved fine-tuning a pre-trained YOLOv5 model on our custom dataset. The final model was trained for 50 epochs and achieved an mAP of 0.95 on the validation set.

#### Inference
To use the model for inference, run the detect.py script with the path to the image you want to detect clocks in, like so:

```
python detect.py --source /path/to/image.jpg
``` 
The script will output an image with bounding boxes around detected clocks.

### Performance
The model achieved an mAP of 0.95 on the validation set, indicating strong performance on the task of clock detection. However, performance may vary depending on the specific use case and image conditions.

### Acknowledgements
YOLOv5 implementation in PyTorch: https://github.com/ultralytics/yolov5
Roboflow platform for dataset creation and annotation: https://roboflow.com/