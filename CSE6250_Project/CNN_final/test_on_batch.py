import pandas as pd
import numpy as np
import os
import sys
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet121
from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model
from my_generator import DataGenerator
from sklearn.metrics import roc_auc_score 
from datetime import datetime


# 0. Set Hyper Parameters.
###################################################################################################
model_name = "VGG16" # must be "VGG16" or "DenseNet121"
if model_name != "VGG16" and model_name != "DenseNet121":
    print("ERROR: Wrong Model Name! Exit Now...")
    sys.exit()

image_csv_list_loc = "../data/dataset_list"
image_file_loc = "../data/images/"
disease_name = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
               "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
batch_size = 32
###################################################################################################


# 1. Import Data
###################################################################################################
# load data split file names.
test_file_list_path = os.path.join(image_csv_list_loc, "test.csv")
test_file_list = pd.read_csv(test_file_list_path)

# generators
test_sequence = DataGenerator(image_file_loc = image_file_loc, disease_name = disease_name, labels = test_file_list, batch_size = batch_size,
                               target_shape = (224, 224), shuffle = False)

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
t1 = datetime.now()
y_pred = model.predict_generator(test_sequence, verbose=1, workers=8)
t2 = datetime.now()

total_test_time = (t2 - t1).total_seconds()
print("Total test time: {} Seconds".format(str(total_test_time)))

y_true = test_sequence.getTrueLabel()
aucList = []
for i in range(len(disease_name)):
    one_auc = roc_auc_score(y_true[:, i], y_pred[:, i])
    aucList.append(one_auc)
    print("{} : {}".format(disease_name[i], one_auc))

print("Mean AUC: {}".format(np.mean(aucList)))
###################################################################################################