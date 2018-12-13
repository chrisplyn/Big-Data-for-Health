import pandas as pd
import numpy as np
import os
import sys
import cv2
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet121
from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model


###################################################################################################
model_name = "VGG16" # must be "VGG16" or "DenseNet121"
if model_name != "VGG16" and model_name != "DenseNet121":
    print("ERROR: Wrong Model Name! Exit Now...")
    sys.exit()

image_name = "00000063_000.png"
image_loc = os.path.join("./test_images/", image_name)
disease_name = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
               "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
true_labels_loc = "./dataset_list/test.csv"
###################################################################################################


# 1. Import Data
###################################################################################################
def readImg(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Note: opencv read BGR rather than RGB, need convert here.
    image = image.astype(np.float64) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return image

image = readImg(image_loc)
image = image.reshape((1, 224, 224, 3))

true_labels = pd.read_csv(true_labels_loc)
y_true = true_labels[true_labels["Image Name"] == image_name].loc[:, disease_name].values
print("Loading Data Done\n")
###################################################################################################


# 2. Build Model 
###################################################################################################
input_tensor = Input(shape=(224, 224, 3))

if model_name == "VGG16":
    base_model = VGG16(include_top=False, weights="imagenet", pooling="avg")
if model_name == "DenseNet121":
    base_model = DenseNet121(include_top=False, weights="imagenet", pooling="avg") 

x = base_model(input_tensor)
predictions = Dense(len(disease_name), activation="sigmoid")(x)
model = Model(inputs=input_tensor, outputs=predictions)

# load model weights, use the path below. 
if model_name == "VGG16":
    weights_path = "./vgg16_weights.h5"
if model_name == "DenseNet121":
    weights_path = "./densenet_weights.h5"

print(model.summary())
print("Building Model Done\n")

model.load_weights(weights_path)
print("Model weight loaded\n")
###################################################################################################


# 3. Testing
###################################################################################################
y_pred = model.predict(image)
print(model_name + " Prediction: \n")
for i in range(len(disease_name)):
    print("{} : {}".format(disease_name[i], y_pred[0, i]))

if len(y_true) == 1:
    print("\nTrue labels: \n")
    for i in range(len(disease_name)):
        print("{} : {}".format(disease_name[i], y_true[0, i]))
###################################################################################################