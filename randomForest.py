import pandas as pd
import numpy as np
import os
from sklearn import linear_model, svm
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectPercentile, f_classif, chi2
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from scipy import stats
# from scipy.stats import gaussian_kde
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import graphviz
from graphviz import Digraph

from pylab import scatter, show, legend, xlabel, ylabel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import plot, draw, show
  

def read_data(path):
	data = np.load(path)
	column_name = data[0].decode('UTF-8').strip().split(',')
	data_list = []
	for train in data[1:]:
		tmp = train.decode('UTF-8').split(',')
		tmp = [0 if t == 'NA' else float(t) for t in tmp]
		data_list.append(tmp)

	df = pd.DataFrame(data=data_list, columns=column_name)

	return df

def remove_low_variance(thre, X):
	sel = VarianceThreshold(threshold=(thre * (1 - thre)))
	X_new = sel.fit_transform(X)
	label = np.array([ i for i in sel.get_support(indices = True )])
	return X_new, label

def univariate_feature(kn, X): # [278 319 401 402 619 666 755 756 757 758]
	sel = SelectKBest(f_classif, k = kn)
	y = X[:, -1]
	X2 = X[:, :-1]
	sel.fit(X2, y)
	
	label = np.array([i for i in sel.get_support(indices = True)])
	X_new = X[:, label]
	X_new = np.c_[X_new, y]
	return X_new,label 

def PCA(X, k = 10):
	# Min-Max Scaler
	scaler = preprocessing.MinMaxScaler()
	y = X[:, -1]
	X_scaler = scaler.fit_transform(X[:, :-1])
	# X_scaler = np.c_[X_scaler, y]

	pca = decomposition.PCA(n_components = k)
	X_scaler = pca.fit_transform(X_scaler)
	# draw(X_scaler, y)
	return X_scaler, y

def predict_curve(y_test, y_pre, title = "predict_curve"):
	plt.figure()
	lines = np.rec.fromarrays([y_test, y_pre])
	lines.sort()
	# x_ax = range(0,1000)
	# plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
	plt.plot(range(0,lines.f0.shape[0]), lines.f0,  color="black", c="darkorange", label="data", linewidth= 2)
	plt.plot(range(0,lines.f1.shape[0]), lines.f1, color="cornflowerblue", label="predict", linewidth=2)
	# plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
	plt.xlabel("data")
	plt.ylabel("target")
	plt.title(title)
	plt.legend()
	plt.show()


def main():

	# df = pd.DataFrame(data=data_list, columns=column_name)
	# [id, 769 features, loss]
	if os.path.exists('train.df'):
		train_df = pd.read_pickle('train.df')
	else:
		train_df = read_data("ecs171train.npy")
		train_df.to_pickle('train.df')
	
	# seperate default and non default data, np.array
	train_data = np.array(train_df)[:, 1: ] # remove id
	# draw_density(train_data)
	default_data = train_data[train_data[:, -1] > 0.5] # (4626, 770)
	# default_data = np.c_[default_data[:, :-1], np.ones(default_data.shape[0])] # classify to one
	nondefault_data = train_data[train_data[:, -1] <= 0.5] # # (45374, 770)

	# PCA_plot2d(default_data, nondefault_data) # by 3 features from PCA
	# return 

	# duplicate to balance
	# for i in range(3): # 3, 0.6; 6, 0.866 (256078, 758)
	#     default_data = np.append(default_data, default_data, axis = 0)
	# train8w = np.append(default_data, nondefault_data[:1000, :], axis = 0) #(82382, 770)
	# for svm with class weight

	train8w = default_data # try to use tree to predict value
	# train8w = np.append(default_data, nondefault_data, axis = 0)  
	np.random. shuffle(train8w)
	# univ_train8w, label_univ = univariate_feature(10, train8w)

	var_train8w, label_lowVar = remove_low_variance(1, train8w) # all the same value
	# print var_train8w.shape # (82382, 570) 0.8; 588 0.9; 1.0 759
	# print label_lowVar

	X, y = PCA(var_train8w, 20)
	# decision_tree_classifier(X,y) # 1 vs 0, 0.826300; 100 - 0; 0.813400
	# decision_tree_regression(X,y) # 1 vs 0, 0.826300; 100 - 0; 0.813400

	# X = univ_train8w[:, :-1]
	# y = univ_train8w[:, -1] 
	# X = var_train8w[:, :-1]
	# y = var_train8w[:, -1]

	skilearn_linear(X,y)
	# skilearn_logistic(X,y)
	# skilearn_svm(X,y)

	# visualize data, uncomment "show()" to run it
	# pos = np.where(y == 1)
	# neg = np.where(y == 0)
	# scatter(X[pos, 4], X[pos, 9], marker='o', c='b')
	# scatter(X[neg, 4], X[neg, 9], marker='x', c='r')
	# xlabel('X1')
	# ylabel('X2')
	# show()

   

if __name__ == '__main__':
	main()
