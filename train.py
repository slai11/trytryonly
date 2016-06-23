# _*_ coding: utf-8 _*_
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics, cross_validation
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

from model import *
from features import *
import warnings
warnings.filterwarnings("ignore")



def search_grid(X, y):
	model = svc_model()
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=10)
	param_grid = dict(svc__kernel=["rbf"], svc__C=[1,5,10,20,50], svc__tol=[1e-8, 1e-4, 1e-2, 1e-1], svc__gamma = np.logspace(-9, 3, 13))

	#dict(xgb__max_depth=np.arange(1,20,1), xgb__gamma=np.arange(0,10,1), xgb__eta=np.arange(0.01, 1, 0.05), xgb__max_delta_step=np.arange(0,100,1))
	#dict(select__percentile=np.arange(1,99,10), svc__kernel=['rbf'], svc__gamma=np.arange(0,10,1), svc__C = (2.0**np.arange(-10, 10, 4)), svc__tol=[1e-8, 1e-4, 1e-1])
	#dict(select__percentile=np.arange(10,99,10), extra__n_estimators=np.arange(1,100,10), extra__max_features=[None, 'auto', 'sqrt', 'log2'])
	#dict(select__percentile=np.arange(10,99,10), randf__max_features=["auto", "log2", None], randf__n_estimators=np.arange(1,100,10), randf__min_samples_split=np.arange(1,100,10), randf__max_depth=[None, 1, 10, 20, 30])
	#dict(select__percentile=np.arange(1,99,10), gb__loss=['deviance', 'exponential'], gb__learning_rate = (2.0**np.arange(-10,10,4)), gb__max_depth= np.arange(1,20,2), gb__max_features=[None, "auto", "sqrt", "log2"])
	#dict(select__percentile=np.arange(1,99,10), knear__algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'], knear__n_neighbors=np.arange(1,20,1), knear__weights=['uniform', 'distance'])
	#dict(select__percentile=np.arange(1,99,10), linsvc__C=(2.0**np.arange(-10,10,4)), linsvc__penalty=['l1','l2'], linsvc__tol=[1e-8, 1e-4, 1e-2, 1e-1])
	#dict(select__percentile=np.arange(10,99,10), randf__max_features=["auto", "log2", None], randf__n_estimators=np.arange(1,100,10), randf__min_samples_split=np.arange(1,100,10), randf__max_depth=[None, 1, 10, 20, 30])
	#dict(select__percentile=np.arange(1,99,10), logre__penalty=['l1', 'l2'], logre__C = (2.0**np.arange(-10, 20, 4)), logre__tol=[1e-8, 1e-4, 1e-1])
	grid = GridSearchCV(model, cv=5, param_grid=param_grid, scoring='f1_macro', n_jobs=-1, verbose=4)
	grid.fit(X,y)

	for i in grid.grid_scores_:
		print i

	print "Best params: " + str(grid.best_params_)
	print "Best score: " + str(grid.best_score_)

if __name__ == '__main__':
	X, y = get_X_y()
	print "X and y are extracted"
	print X.shape # for checking

	#search_grid(X,y)
	
	clf = lin_svc_model()

	
	print "Done, time to cross-validate"
	scores = cross_validation.cross_val_score(clf, X, y, cv = 5, scoring = "accuracy", n_jobs = 2, verbose=2)
	print scores
	

	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
	
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