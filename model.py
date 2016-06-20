# _*_ coding: utf-8 _*_
from PIL import Image, ImageFilter
import PIL
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score, precision_recall_curve, average_precision_score
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC

from features import *

def log_model():
	clf = LogisticRegression(class_weight = "balanced", penalty='l1')
	scaler = MinMaxScaler()
	select = SelectPercentile(score_func=chi2, percentile=10)
	pipeline = Pipeline([('scale', scaler), ('select', select), ('logre', clf)])
	return pipeline

def svc_model():
	clf = LinearSVC()
	scaler = MinMaxScaler()
	select = SelectPercentile(score_func=chi2, percentile=10)
	pipeline = Pipeline([('scale', scaler), ('select', select), ('logre', clf)])
	return pipeline

if __name__ == '__main__':
	X, y = get_X_y()
	print "X and y are extracted"
	print X.shape # for checking

	clf = log_model()

	"""
	print "Done, time to cross-validate"
	scores = cross_validation.cross_val_score(clf, X, y, cv = 5, scoring = "accuracy", n_jobs = 2, verbose=2)
	print scores
	"""

	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80, random_state=10)
	
	print X_test.shape
	clf.fit(X_train, y_train)
	print "fitted"
	y_pred = clf.predict(X_test)
	print "Classification Report:"
	print metrics.classification_report(y_test, y_pred)
	print f1_score(y_test, y_pred, average='macro')
	cm = confusion_matrix(y_test, y_pred)
	
	print "Confusion Matrix:"
	print cm




