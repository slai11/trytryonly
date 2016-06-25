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
	#### Uncomment out the model you are testing
	#model = log_model
	#model = lin_svc_model()
	model = svc_model()
	#model = rf_model()
	#model = k_nearest_model()
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=10)

	#### Un-comment the relevant param_grid
	#param_grid = dict(logre__penalty=['l1', 'l2'], logre__C = [1,5,10,20,50], logre__tol=[1e-8, 1e-4, 1e-1])
	#param_grid = dict(linsvc__C=[1,5,10,20,50], linsvc__penalty=['l1','l2'], linsvc__tol=[1e-8, 1e-4, 1e-2, 1e-1])
	#param_grid = dict(randf__max_features=["auto", "log2", None], randf__n_estimators=np.arange(1,100,20), randf__min_samples_split=np.arange(1,100,10), randf__max_depth=[None, 1, 10, 20, 30])
	param_grid = dict(svc__kernel=['rbf'], svc__gamma=np.arange(0,10,1), svc__C =[1,5,10,20,50], svc__tol=[1e-8, 1e-4, 1e-1])
	#param_grid = dict(knear__algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'], knear__n_neighbors=np.arange(1,20,2), knear__weights=['uniform', 'distance'])

	#param_grid = dict(svc__kernel=["rbf"], svc__C=[1,5,10,20,50], svc__tol=[1e-8, 1e-4, 1e-2, 1e-1], svc__gamma = np.logspace(-9, 3, 13))
	#dict(xgb__max_depth=np.arange(1,20,1), xgb__gamma=np.arange(0,10,1), xgb__eta=np.arange(0.01, 1, 0.05), xgb__max_delta_step=np.arange(0,100,1))
	#dict(select__percentile=np.arange(1,99,10), svc__kernel=['rbf'], svc__gamma=np.arange(0,10,1), svc__C = (2.0**np.arange(-10, 10, 4)), svc__tol=[1e-8, 1e-4, 1e-1])	

	grid = GridSearchCV(model, cv=4, param_grid=param_grid, scoring='f1_macro', n_jobs=-1, verbose=4)
	grid.fit(X,y)

	for i in grid.grid_scores_:
		print i

	print "Best params: " + str(grid.best_params_)
	print "Best score: " + str(grid.best_score_)

def grid(X, y, X_sub):
	model = svc_model()
	param_grid = dict(svc__kernel=['rbf'], svc__gamma=np.arange(0,10,1), svc__C =[1,5,10,20,50], svc__tol=[1e-8, 1e-4, 1e-1])
	grid = GridSearchCV(model, cv=4, param_grid=param_grid, scoring='accuracy', n_jobs=-1, verbose=4)
	grid.fit(X,y)

	for i in grid.grid_scores_:
		print i

	print "Best params: " + str(grid.best_params_)
	print "Best score: " + str(grid.best_score_)

	output = grid.predict(X_sub)
	output = pd.DataFrame(output)
	output.to_csv("pred.csv", index=False)

if __name__ == '__main__':
	X, y = get_X_y("id_train.csv")
	print "X and y are extracted"
	print X.shape # for checking

	X_sub, y_sub = get_X_y("sample_submission4.csv")
	#X_sub = pd.DataFrame(X)
	

	grid(X, y, X_sub)

	"""select = SelectPercentile(score_func=chi2, percentile=15)
				X = select.fit_transform(X)
				X = pd.DataFrame(X)
				X.to_csv("featreduced.csv", index=False)
				y = pd.DataFrame(y)
				y.to_csv("label.csv", index=False)"""

	'''clf = k_nearest_model()
			
				clf.fit(X, y)
			
				print "fitted"
			
				
			
			
				y_pred = clf.predict(X_sub)
				print "predicted"
			
				out = pd.DataFrame(y_pred)
				out.to_csv("out3.csv")
			'''

	
