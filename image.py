import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D




"""
probably have to do something like this -> 
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
1      3479
2      1856
3       859
4      1806
"""

# dimension of images not consistent -> resize to

image_height = 200
image_width = 200

train_data_dir = 'roof_images/train'
validation_data_dir = 'roof_images/validation'
submission_data_dir = 'roof_images/submission'


def gen_train_test():
	# assigns the train and validation set to class folders
	# randomly assigns 20% of train_id to validation
	# assigns the rest 
	id_ = pd.read_csv("id_train.csv")
	tuplist = id_[["Id", "label"]]
	newlist = [tuple(x) for x in tuplist.values]

	newfilepath1 = 'roof_images/train/1/'
	newfilepath2 = 'roof_images/train/2/'
	newfilepath3 = 'roof_images/train/3/'
	newfilepath4 = 'roof_images/train/4/'
	valfilepath1 = 'roof_images/validation/1/'
	valfilepath2 = 'roof_images/validation/2/'
	valfilepath3 = 'roof_images/validation/3/'
	valfilepath4 = 'roof_images/validation/4/'

	# splitting test cases into individual 
	for tup in newlist:
		oldfilepath = train_data_dir + tup[0] + ".jpg"
		if int(tup[1]) is 1:
			#shift to 1
			newfilepath = newfilepath1 + tup[0] + ".jpg"
		elif int(tup[1]) is 2:
			newfilepath = newfilepath2 + tup[0] + ".jpg"
		elif int(tup[1]) is 3:
			newfilepath = newfilepath3 + tup[0] + ".jpg"
		else:
			newfilepath = newfilepath4 + tup[0] + ".jpg"

		os.rename(oldfilepath, newfilepath)

	# time to random split out 1600 into indiv folders
	


	




model = Sequential()
model.add(Convolution2D(32,3,3),input_shape=(3, image_width, image_height))
model.add(MaxPooling2D(pool_size=(2,2)))








