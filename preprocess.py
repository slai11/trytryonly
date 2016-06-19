import numpy as np
import pandas as pd
import os



"""
# split the data into 
1. training data  (make use of this well!)
2. submission data
3. others

"""

def sort_files():
	id_ = pd.read_csv("id_train.csv")
	sub = pd.read_csv("sample_submission4.csv")

	path = "roof_images/train/"
	path2 = "roof_images/submission/"

	id_list = [id_['Id']]
	sub_list = [sub['Id']]

	for pic in id_list[0]:
		pic = str(pic)
		oldname = "roof_images/" + pic + ".jpg"
		newname =  path + pic + ".jpg"
		os.rename(oldname, newname)

	for pic in sub_list[0]:
		pic = str(pic)
		oldname = oldname = "roof_images/" + pic + ".jpg"
		newname = path2 + pic + ".jpg"
		os.rename(oldname, newname)

