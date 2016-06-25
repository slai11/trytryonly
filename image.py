import numpy as np
import pandas as pd
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.models import Sequential


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
1      3479 - 2784 695
2      1856 - 1485 371
3       859 - 688 171
4      1806 - 1443 363
"""

# dimension of images not consistent -> resize to

image_height = 200
image_width = 200

train_data_dir = 'roof_images/train/'
validation_data_dir = 'roof_images/validation/'
submission_data_dir = 'roof_images/submission/'


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

	one = 0
	two = 0
	three = 0
	four = 0
	# splitting test cases into individual 
	for i, tup in enumerate(newlist):
		pic = str(tup[0])
		oldfilepath = train_data_dir + pic + ".jpg"
		if int(tup[1]) is 1:
			if one < 2784:
				newfilepath = newfilepath1 + pic + ".jpg"
			else:
				newfilepath = valfilepath1 + pic + ".jpg"
			one+=1
		elif int(tup[1]) is 2:
			if two < 1485:
				newfilepath = newfilepath2 + pic + ".jpg"
			else:
				newfilepath = valfilepath2 + pic + ".jpg"
			two+=1
		elif int(tup[1]) is 3:
			if three < 688:
				newfilepath = newfilepath3 + pic + ".jpg"
			else:
				newfilepath = valfilepath3 + pic + ".jpg"
			three+=1
		else:
			if four < 1443:
				newfilepath = newfilepath4 + pic + ".jpg"
			else:
				newfilepath = valfilepath4 + pic + ".jpg"
			four+=1

		os.rename(oldfilepath, newfilepath)
	

if __name__ == '__main__':
	#gen_train_test()

	model = Sequential()
	model.add(ZeroPadding2D((1, 1), input_shape=(3, image_width, image_height)))
	model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	'''model.add(ZeroPadding2D((1, 1)))
				model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
				model.add(ZeroPadding2D((1, 1)))
				model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
				model.add(ZeroPadding2D((1, 1)))
				model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
				model.add(MaxPooling2D((2, 2), strides=(2, 2)))
			
				model.add(ZeroPadding2D((1, 1)))
				model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
				model.add(ZeroPadding2D((1, 1)))
				model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
				model.add(ZeroPadding2D((1, 1)))
				model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
				model.add(MaxPooling2D((2, 2), strides=(2, 2)))'''

	'''
	model.add(Convolution2D(32, 3, 3, input_shape=(3, image_width, image_height)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(32, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	'''

	model.add(Flatten())
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('softmax'))

	model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


	train_datagen = ImageDataGenerator(rescale=1./255, \
										shear_range=0.2,\
										zoom_range=0.2,\
										horizontal_flip=True)

	test_datagen= ImageDataGenerator(rescale=1./255)

	train_gen = train_datagen.flow_from_directory(train_data_dir, \
												target_size=(image_width,image_height),\
												batch_size=32,\
												class_mode='binary')

	test_gen = test_datagen.flow_from_directory(validation_data_dir, \
												target_size=(image_width,image_height),\
												batch_size=32,\
												class_mode='binary')

	model.fit_generator(train_gen,\
				samples_per_epoch=6400,\
				nb_epoch=20,\
				validation_data=test_gen,\
				nb_val_samples=1600)


	model.save_weights('first_try.h5')




