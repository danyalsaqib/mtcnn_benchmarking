# mtcnn_recognition
The link to the arcface ONNX model used can be found at: https://drive.google.com/file/d/16aJ_uiDWeggv0V7i9VBSzrdIOUhK84Qr/view?usp=sharing

This model should be placed in 'face_recognition/models/'.

## Dockerfile
The dockerfile contains instructions to first install necessary libraries, and then copy the 'retinaface_functions/' directory into a local docker directory using the command `COPY face_recognition /face_recognition`. The docker uses the tensorflow container as its parent, given by `FROM nvcr.io/nvidia/tensorflow:21.11-tf2-py3`

To build the docker, simply use the following command. Remember that the dot at the end is important:
```
sudo docker build -t mtcnn_pipeline:single .
```
To run the docker after successfully building it, use the following command:
```
sudo docker run --gpus all -it --rm mtcnn_pipeline:single
```

## Running the complete pipeline
After running the docker, change into the following directory:
```
cd /face_recognition/
````

The 'python3 face_main.py/face_main.py' file has a main functon, that can run the entire pipeline with 1 call. Its usage is as follows:
```
python3 face_main.py
```

As an example, by default, the inference runs on a sample image. The inference is essentially done by calling the infer_image function, and passing it a path to an image. As an example, it runs:
```
test_pic = "Aamir Hussain Liaquat_img_1.jpg"
infer_image(test_pic)
```
