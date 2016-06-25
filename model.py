# _*_ coding: utf-8 _*_
from PIL import Image, ImageFilter
import PIL
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import metrics, cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score, precision_recall_curve, average_precision_score
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier, BaggingClassifier

from features import *

def log_model():
	clf = LogisticRegression(class_weight = "balanced", penalty='l1')
	scaler = MinMaxScaler()
	select = SelectPercentile(score_func=chi2, percentile=10)
	pipeline = Pipeline([('scale', scaler), ('select', select), ('logre', clf)])
	return pipeline

def svc_model():
	clf = SVC(class_weight='balanced', kernel='rbf', C=10)
	scaler = MinMaxScaler()
	select = SelectPercentile(score_func=chi2, percentile=10)
	pipeline = Pipeline([('scale', scaler), ('select', select), ('svc', clf)])
	return pipeline

def lin_svc_model():
	clf = LinearSVC(class_weight='balanced', C=10)
	scaler = MinMaxScaler()
	select = SelectPercentile(score_func=chi2, percentile=10)
	pipeline = Pipeline([('scale', scaler), ('select', select), ('linsvc', clf)])
	return pipeline	

def rf_model():
	clf = RandomForestClassifier(class_weight='balanced')
	scaler = MinMaxScaler()
	select = SelectPercentile(score_func=chi2, percentile=1)
	pipeline = Pipeline([('scale', scaler), ('select', select), ('ranf', clf)])
	return pipeline

def k_nearest_model():
	select = SelectPercentile(score_func=chi2, percentile=10)
	knc = KNeighborsClassifier(weights='uniform', algorithm='auto', n_neighbors = 15)
	scaler = MinMaxScaler()
	pipeline = Pipeline([('scale', scaler), ('select', select) , ('knear', knc)])
	return pipeline

def xgb_model():
	clf = xgb.XGBClassifier()
	scaler = MinMaxScaler()
	select = SelectPercentile(score_func=chi2, percentile=10)
	pipeline = Pipeline([('scale', scaler), ('select', select), ('xgb', clf)])
	return pipeline

def sgd_model():
	clf = SGDClassifier(class_weight='balanced')
	scaler = MinMaxScaler()
	select = SelectPercentile(score_func=chi2, percentile=50)
	pipeline = Pipeline([('scale', scaler), ('select', select), ('sgd', clf)])
	return pipeline


