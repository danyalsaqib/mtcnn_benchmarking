import cv2
import numpy as np
import pandas as pd
import json
import os
import time
from mtcnn_ort import MTCNN
from utils import *
import onnxruntime

# Initializing the detector
detector = MTCNN()

# Initializing the recognizer
model_path = "models/arcface.onnx"
session = onnxruntime.InferenceSession(model_path, None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)
src[:, 0] += 8.0

tform = trans.SimilarityTransform()


"""
def face_infer(recieved_image_path):
    # images_path is the list containing 
    images_path = json.load(recieved_image_path)
    for image_path in images_path[]:
"""

def infer_image(img_path):

    image = cv2.imread(img_path)
    print(str(type(image))[8:-2])
    if (str(type(image))[8:-2] == 'NoneType'):
        print("Invalid Image")
        return -1
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces_raw(image)
    #detection_results_show(img_path, faces)

    for i in range(len(faces[0])):
        # Size of Image is less than 100 pixel,scrap it
        image = cv2.imread(img_path)
        if ((int(faces[0][i][3]) - int(faces[0][i][1])) > 100 or (int(faces[0][i][2]) - int(faces[0][i][0])) > 100):
            crop_image = image[int(faces[0][i][1]): int(faces[0][i][3]), int(faces[0][i][0]):int(faces[0][i][2])]
            landmarks = [int(faces[1][0][i]), int(faces[1][1][i]), int(faces[1][2][i]), int(faces[1][3][i]), int(faces[1][4][i]),
                        int(faces[1][5][i]), int(faces[1][6][i]), int(faces[1][7][i]), int(faces[1][8][i]), int(faces[1][9][i])]
            if find_roll(landmarks) > - 20 and  find_roll(landmarks) < 20 and find_yaw(landmarks) > -50 and find_yaw(landmarks) < 50 and find_pitch(landmarks) < 2 and find_pitch(landmarks) > 0.5:
                #cropped_results_show(crop_image, landmarks)
                # Recognition
                facial5points = np.reshape(landmarks, (2, 5)).T
                tform.estimate(facial5points, src)
                M = tform.params[0:2, :]
                img = cv2.warpAffine(image, M, (112, 112), borderValue=0.0)
                #cropped_results(img)
                #print(img.shape)
                blob = cv2.dnn.blobFromImage(img, 1, (112, 112), (0, 0, 0))
                blob -= 127.5
                blob /= 128
                result = session.run([output_name], {input_name: blob})
                dictionary = {'names': "bajwa sb", 'embeddings': result[0][0].tolist()}

                return result[0][0]

                #with open('dictionary.json', 'w') as handle:
                #    json.dump(dictionary, handle)

                #distance = findCosineDistance(result[0][0], bajwa)
                #print(distance)


            else:
                print("Invalid Pose")
                return -1
        else:
            print("Image size is small")
            return -1

def findCosineDistance(source_representation, test_representation):
    #print("Source Representation: ", source_representation.shape)
    #print("Test Representation: ", test_representation.shape)
    a = np.matmul(np.transpose(source_representation), test_representation)
    print("a: ", a.shape)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    x = 1 - (a / (np.sqrt(b) * np.sqrt(c)))
    return x[0][0]

if __name__ == '__main__':
    detector = MTCNN()
    count = 0
    #test_pic = "images/1_4.jpg"
    #infer_image(test_pic)
    file_name = 'benchmarking.csv'
    out_df = pd.DataFrame(columns=['Source Image', 'Target Image', 'Distance'])
    out_df.to_csv(file_name)
    iter_df = pd.DataFrame(columns=['Source Image', 'Target Image', 'Distance'])
    #hm = get_representation('images/3.5cca991308724.jpg')
    start_time = time.time()
    for subdir, dirs, files in os.walk(os.path.join(os.getcwd(), 'test')):
        for file_names_1 in files:
            print("Source Image: ", file_names_1)
            im1 = infer_image(os.path.join(os.getcwd(), 'test', file_names_1))
            count += 1
            if (str(type(im1))[8:-2] != 'NoneType') and np.sum(im1) != -1 and len(im1) == 512:
                im1 = np.array(im1)
                im1 = im1.reshape(512, 1)
                for subdir, dirs, files in os.walk(os.path.join(os.getcwd(), 'test')):
                    for file_names_2 in files:
                        print("Target Image: ", file_names_2)
                        im2 = infer_image(os.path.join(os.getcwd(), 'test', file_names_2))
                        count += 1
                        print("Iteration: ", count)
                        if (str(type(im2))[8:-2] != 'NoneType') and np.sum(im2) != -1 and len(im2) == 512:
                            im2 = np.array(im2)
                            im2 = im2.reshape(512, 1)
                            dist_im = findCosineDistance(im1, im2)
                            print("Distance: ", dist_im)
                            val_out = [str(file_names_1), str(file_names_2), dist_im]
                            iter_df.loc[0] = val_out
                            iter_df.to_csv(file_name, mode='a', header=False)
                            print("Written to File")
    end_time = time.time()
    print("Start Time: ", start_time)
    print("End Time: ", end_time)
    print("Total Time: ", end_time - start_time)
    print("Total inferences: ", count)
    print("Time/Inference: ", ((end_time - start_time) / (count)))
