import pandas as pd
import numpy as np
import os
from sklearn import linear_model, svm
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectPercentile, f_classif, chi2
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.decomposition import PCA
from sklearn import decomposition
from sklearn.metrics import average_precision_score
from scipy import stats
# from scipy.stats import gaussian_kde
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import export_graphviz
import graphviz
from graphviz import Digraph

from pylab import scatter, show, legend, xlabel, ylabel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import plot, draw, show

# Mar 15
# use all feature
# duplicate more 1 will increase accuracy, 3 - 0.610992425714; 6 - 0.866096532334; 5 - 0.766255790867
# use selectKBest(10)
# duplicate more 1 will increase accuracy, 3 - 0.606476985822; 6 - 0.866084817245; 5 - 0.768861681006


def draw(X, y):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	n = 100

	# For each set of style and range settings, plot n random points in the box
	# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
	for yax in y:
		# xs = randrange(n, 23, 32)
		# ys = randrange(n, 0, 100)
		# zs = randrange(n, zlow, zhigh)
		# ax.scatter(xs, ys, zs, c=c, marker=m)
		ax.scatter(X[:,0], X[:,1],y, c= yax, marker = 'r')

	ax.set_xlabel('X0 Label')
	ax.set_ylabel('X1 Label')
	ax.set_zlabel('Y Label')

	plt.show()
def draw_density(train_data):
	y_den = train_data[:, -1]
	y_den = [y for y in y_den if y > 0.5 ]
	print y_den
	x_den = np.arange(1,101,1)
	density = stats.kde.gaussian_kde(y_den)
	plt.plot(x_den, density(x_den))
	plt.show()
	plt.close()

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

def PCA_plot3d(X1, X2, k = 3):
	# Min-Max Scaler
	scaler = preprocessing.MinMaxScaler()
	X1_scaler = scaler.fit_transform(X1[0:100, :-1])
	X2_scaler = scaler.fit_transform(X2[0:800, :-1])
	# X_scaler = np.c_[X_scaler, y]

	pca = decomposition.PCA(n_components = k)
	X1_scaler = pca.fit_transform(X1_scaler)
	X2_scaler = pca.fit_transform(X2_scaler)

	fig = plt.figure(figsize = (8,8))
	ax = fig.add_subplot(111, projection = '3d')
	plt.rcParams['legend.fontsize'] = 10
	ax.plot(X1_scaler[:, 0], X1_scaler[:, 2], X1_scaler[:, 2], 'o', markersize=8, color='blue', alpha=0.5, label='class1')
	ax.plot(X2_scaler[:, 0], X2_scaler[:, 1], X2_scaler[:, 2], '^', markersize=8, alpha=0.5, color='red', label='class0')
	plt.title('Samples for class 1 and class 0')
	ax.legend(loc = 'upper right')
	plt.show()
	return 

def PCA_plot2d(X1, X2, k = 2):
	# Min-Max Scaler
	scaler = preprocessing.MinMaxScaler()
	X1_scaler = scaler.fit_transform(X1[0:100, :-1])
	X2_scaler = scaler.fit_transform(X2[0:800, :-1])
	# X_scaler = np.c_[X_scaler, y]

	pca = decomposition.PCA(n_components = k)
	X1_scaler = pca.fit_transform(X1_scaler)
	X2_scaler = pca.fit_transform(X2_scaler)

	plt.plot(X1_scaler[0:100,0],X1_scaler[0:100,1], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
	plt.plot(X2_scaler[0:100,0], X2_scaler[0:100,1], '^', markersize=7, color='red', alpha=0.5, label='class0')

	plt.xlabel('x1_values')
	plt.ylabel('x2_values')
	# plt.xlim([-4,4])
	# plt.ylim([-4,4])
	plt.legend()
	plt.title('Transformed samples with class labels from matplotlib.mlab.PCA()')

	plt.show()
	return 

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

def skilearn_linear(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
	print 'train size: ', X_train.shape
	print 'test size: ', X_test.shape

	logreg = linear_model.LinearRegression() #, max_iter=1000, solver = 'sag')
	logreg.fit(X_train, y_train)
	accuracy = logreg.score(X_test, y_test)

	y_pre = logreg.predict(X_test)
	y_pre = np.round(y_pre,0)
	print 'predict #: ',y_pre.shape[0]
	print 'predict non zero #: ', np.count_nonzero(y_pre)
	print 'skilearn_logistic scores #: ', accuracy # without balance data, 10 feature, skilearn_svm scores #:  0.9076; skilearn_svm scores with class weight#:  0.62568
	
	predict_curve(y_test, y_pre, "linear regression")

def skilearn_logistic(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
	print 'train size: ', X_train.shape
	print 'test size: ', X_test.shape

	logreg = linear_model.LogisticRegression(C = 10) #, max_iter=1000, solver = 'sag')
	logreg.fit(X_train, y_train)
	accuracy = logreg.score(X_test, y_test)

	y_pre = logreg.predict(X_test)
	print 'predict #: ',y_pre.shape[0]
	print 'predict non zero #: ', np.count_nonzero(y_pre)
	print 'skilearn_logistic scores #: ', accuracy # without balance data, 10 feature, skilearn_svm scores #:  0.9076; skilearn_svm scores with class weight#:  0.62568


def decision_tree_classifier(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	print 'train size: ', X_train.shape
	print 'test size: ', X_test.shape
	tree = DecisionTreeClassifier(max_depth = 15, random_state=0)
	tree.fit(X_train, y_train)
	y_pre = tree.predict(X_test)
	y_pre = np.round(y_pre,0)
	# print("accuracy on training set: %f" % tree.score(X_train, y_train))
	print("accuracy on test set: %f" % tree.score(X_test, y_test))
	print 'non zero predict:', np.count_nonzero(y_pre) 
	print 'non zero test:', np.count_nonzero(y_test) 

	predict_curve(y_test, y_pre, "decision_tree_classifier predict curve")

	# feature importance
	# plt.plot(tree.feature_importances_, 'o')
	# plt.xticks(range(X_train.shape[1]), range(0,20))
	# plt.ylim(0, 0.5)
	# plt.show()

	# error
	# if not os.path.exists('mytree.dot'):
	# 	export_graphviz(tree, out_file="mytree.dot", impurity=False, filled=True)
	# with open("mytree.dot") as f:
	# 	dot_graph = f.read()
	# 	graph = graphviz.Source(dot_graph)
	# 	graph.render()
	# dot = Digraph(comment='The Round Table')
	# dot.render('Source.gv', view=True)  
def decision_tree_regression(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	print 'train size: ', X_train.shape
	print 'test size: ', X_test.shape
	tree = DecisionTreeRegressor(max_depth = 8)
	tree.fit(X_train, y_train)
	y_pre = tree.predict(X_test)
	# y_pre = np.round(y_pre,0) 
	# print("accuracy on training set: %f" % tree.score(X_train, y_train))
	print("accuracy on test set: %f" % tree.score(X_test, y_test))
	print 'non zero predict:', np.count_nonzero(y_pre) 
	print 'non zero test:', np.count_nonzero(y_test) 

	predict_curve(y_test,y_pre,"decision_tree_regression predict curve")

def skilearn_svm(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
	print("train sets: {0}; test sets: {1}".format(X_train.shape, X_test.shape))
	print 'non zero label:',np.count_nonzero(y_test)
	# fit the model and get the separating hyperplane

	# clf = svm.SVC(kernel='linear', C=1.0)
	# clf.fit(X_train, y_train)
	# y_pre = clf.predict(X_test)
	# score = clf.score(X_test, y_test)
	# print 'skilearn_svm scores #: ', score
	# print 'non zero predict: ', np.count_nonzero(y_pre)
	try:
		W = []
		Score = []
		Y_pre = []
		average_precision_score
		for w in range(1,10):
			# wclf = svm.SVC(kernel='linear',class_weight={1: w})
			wclf = svm.LinearSVC(class_weight={1: w})
			wclf.fit(X_train, y_train)
			y_pre = wclf.predict(X_test)
			y_score = wclf.score(X_test, y_test)

			Y_pre.append(y_pre)
			W.append(w)
			Score.append(y_score)
			print 'non zero predict:', np.count_nonzero(y_pre)            
			print 'predict scores: ', y_score

		# plot1 = plt.figure()
		# plt.plot(W, Score, 'b-o')
		# plt.show()

		 
	except KeyboardInterrupt:
		pass

	return
   
	# fit the model and get the separating hyperplane using weighted classes
	wclf = svm.SVC(kernel='linear', class_weight={1: 10})
	wclf.fit(X_train, y_train)
	y_pre = wclf.predict(X_test)
	score = wclf.score(X_test, y_test)
	print 'skilearn_svm scores with class weight#: ', score
	print 'non zero predict: ', np.count_nonzero(y_pre)


def randrange(n, vmin, vmax):
	'''
	Helper function to make an array of random numbers having shape (n, )
	with each number distributed Uniform(vmin, vmax).
	'''
	return (vmax - vmin)*np.random.rand(n) + vmin



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
