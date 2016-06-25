# _*_ coding: utf-8 _*_
from PIL import Image, ImageFilter
import PIL
import pandas as pd
import numpy as np

from skimage import feature


"""
Possible new features
http://scikit-image.org/docs/dev/auto_examples/plot_canny.html

canny edge n sobel edge

http://home.in.tum.de/~aichert/html-data/featurepaper.pdf
http://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=features

1. sobel edge
2. canny edge
3. detect blob
4. locate peaks

"""



def get_image_array(image_name):
	"""
	This method opens an image and generates the relevant features before flattening it

	@params: image_name
					file name, string
	@returns image_feat
					(1, 2 * basewidth^2) numpy array
	"""
	filename = "Archive/roof_images/" + str(image_name) + ".jpg"
	
	image_feat = []
	basewidth = 150 #can change this, but best nt to overfit it
	img = Image.open(filename)

	# first feature, convert to black and white
	img = img.convert('L')
	img = img.resize((basewidth, basewidth), PIL.Image.ANTIALIAS)
	array = np.array(img)
	newlength = basewidth*basewidth
	newarray = array.reshape(1, newlength)

	# second feature, apply gaussian blur
	blurred_img = img.filter(ImageFilter.GaussianBlur(radius=2))
	blur = np.array(blurred_img)
	blur = blur.reshape(1, newlength)

	sobel = sobel_edge(array)

	newlength = sobel.shape[0] * sobel.shape[1]
	print newlength
	sobel = sobel.reshape(1, newlength)

	"""blob = detect_blob(array)
				print blob.shape
				newlength = blob.shape[0] * blob.shape[1]
				blob = blob.reshape(1, newlength)"""

	image_feat.append(newarray)
	image_feat.append(blur)
	image_feat.append(sobel)
	#image_feat.append(blob)

	image_feat = np.hstack(image_feat)

	return image_feat


def get_feature_array(inputlist):
	main_x = []
	for tup in inputlist:
		x = get_image_array(tup[0])
		main_x.append(x)
	main_x = np.vstack(main_x)
	return main_x


def get_X_y(filename):
	df = pd.read_csv(filename)
	subset = df[["Id", "label"]]
	inputlist = [tuple(x) for x in subset.values]
	X = get_feature_array(inputlist)

	#build labels
	y = [y[1] for y in inputlist]
	y = np.array(y)

	return X, y


#############
# Features #
#############



def sobel_edge(img):
	edge = feature.canny(img, sigma=2)
	edge = np.array(edge)
	return edge

def detect_blob(img):
	blob = feature.blob_dog(img, threshold = 0.5, max_sigma = 20)
	blob = np.array(blob)
	return blob


