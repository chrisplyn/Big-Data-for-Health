from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model
from django.core.cache import cache
from keras import backend as K
from .xrayConstants import class_names
import numpy as np
import urllib
import cv2

target_shape=(224,224)

def handle_uploaded_file(event):
	work_on_prediction(event)
	pass

def work_on_prediction(event):
	# Check cache model first
	# x_ray_model_key = 'model_cache'
	# model = cache.get(x_ray_model_key)
	# if model is None:
	model = build_model()
	image_input = readImg(event.xray_image.path)
	result = model.predict(np.array([image_input]))
	K.clear_session()
	print("prediction result---------------------->")
	print(result)
	if result is not None:
		pre_result = [round(x*100, 4) for x in result[0].tolist()]
		predictions = list(zip(class_names, np.array(pre_result)))
		predictions = sorted(predictions, key=lambda x: x[1], reverse = True)[0:5]
		event.set_pre_result(predictions)
		event.save()
		return True
	else:
		return False

def build_model():
	input_tensor = Input(shape=(224, 224, 3))
	base_model = VGG16(include_top=False, weights="imagenet", pooling="avg") # Also check "max" pooling
	x = base_model(input_tensor)
	predictions = Dense(len(class_names), activation="sigmoid")(x)
	model = Model(inputs=input_tensor, outputs=predictions)

	# If load previous model weights, use the path below. 
	weights_path = 'static/documents/model.h5'
	if weights_path != "":
	    model.load_weights(weights_path)

	print(model.summary())
	print("Building Model Done\n")
	return model;

def readImg(path=None, stream=None, url=None):

	# read image
	if path is not None:
		image = cv2.imread(path);
	else:
		if url is not None:
			response = urllib.request.urlopen(url)
			image_data = response.read();
		elif stream is not None:
			image_data = stream.read()
			image_content = np.asarray(bytearray(image_data), dtype = 'uint8')
			image = cv2.imdecode(image_content, cv2.IMREAD_GRAYSCALE)

	# process image
	image = cv2.resize(image, target_shape, interpolation = cv2.INTER_CUBIC)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = image.astype(np.float64) / 255.0
	image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
	return image
