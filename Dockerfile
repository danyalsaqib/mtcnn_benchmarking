FROM nvcr.io/nvidia/tensorflow:21.11-tf2-py3

RUN apt-get update && \
    pip install opencv-python && \
    apt-get install ffmpeg libsm6 libxext6  -y && \
    python -m pip install -U scikit-image && \
    pip install numpy protobuf==3.16.0 && \
    pip install onnx && \
    pip install onnxruntime-gpu==1.9.0 && \
    pip install mtcnn-onnxruntime

