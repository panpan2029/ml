import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.model_selection import train_test_split

trainfile = "data1000.csv" #no missing # 
gl = pd.read_csv(trainfile) # 1000 x 772

all1000 = gl.iloc[:, 2: gl.shape[1]] #1000 x 770
all1000 = all1000.as_matrix()
# X = all1000.iloc[:, :769]
# Y = all1000.iloc[:, 769]
# print(type(all1000)) # <type 'numpy.ndarray'>
# print(all1000.shape)
# print(all1000[0])
X = all1000[:, 0:769]
# print(X[0])
Y = all1000[:, 769]
# print(Y[0])



## default is 75% / 25% train-test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
# y_train.reshape(y_train.shape[0],1)
# y_test.reshape(y_test.shape[0],1)
# print(y_train.size)
print("X_train {0}\t, X_test {1}\t, Y_train {2}\t, Y_test{3}\t".format(X_train.shape,X_test.shape,y_train.shape,y_train.shape))
W1 = 2 * np.random.random((X_train.shape[1], 5)) - 1
W2 = 2 * np.random.random((5, 1)) - 1
# print("Y test: {0}".format(y_train[100]))
# yt = np.squeeze(y_train)
# print(yt)
# print("W1 {0}\t, W2 {1}".format(W1.shape,W2.shape))

# http://iamtrask.github.io/2015/07/12/basic-python-network/
np.random.seed(1)

def nonlin(x, deriv = False):
	if(deriv == True):
		return x * (1-x)
	return 1/(1+np.exp(-x))  #sigmoid

# X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
# y = np.array([[0],[1],[1],[0]])

# syn0 = 2*np.random.random((3,4)) - 1
# syn1 = 2*np.random.random((4,1)) - 1

# print("X_train {0}\t, X_test {1}".format(X.shape,y.shape))
# print("syn0 {0}\t, syn1 {1}".format(syn0.shape,syn1.shape))

# for j in xrange(60000):
# 	l0 = X
# 	l1 = nonlin(np.dot(l0,syn0))
# 	l2 = nonlin(np.dot(l1,syn1))

# 	l2_error = y - l2
# 	if (j% 10000) == 0:
# 		print "Error:" + str(np.mean(np.abs(l2_error)))
# 		print("l1 {0}\t, l2 {1}".format(l1.shape,l2.shape))
# 	l2_delta = l2_error*nonlin(l2,deriv=True)
# 	l1_error = l2_delta.dot(syn1.T)
# 	l1_delta = l1_error * nonlin(l1,deriv=True)
# 	syn1 += l1.T.dot(l2_delta)
# 	syn0 += l0.T.dot(l1_delta)

# 	if (j% 10000) == 0:
# 		print("l1_delta {0}\t, l2_delta {1}".format(l1_delta.shape,l2_delta.shape))
# 		print("l1_error {0}\t, l2_error {1}".format(l1_error.shape,l2_error.shape))

syn1 = 2*np.random.random((X_train.shape[1],4)) - 1
syn2 = 2*np.random.random((4,1)) - 1	
for iter in xrange(2):
	# forward propagation
	layer0 = X_train
	layer1 = nonlin(np.dot(layer0, syn1))
	layer2 = nonlin(np.dot(layer1, syn2))
	# how much to miss
	# print(layer2.shape)
	# print(yt.shape)
	layer2_error = y_train - layer2
	print("l1 {0}\t, l2 {1}".format(layer1.shape,layer2.shape))
	# print(layer2_error.shape)
	# print(layer2_error.shape)
	layer2_delta = layer2_error * nonlin(layer2, deriv = True)
	layer1_error = layer2_delta.dot(syn2.T)
	# print(layer1_error.shape)
	layer1_delta = layer1_error * nonlin(layer1, deriv = True)
	# print(layer1.T.shape)
	# print(layer2_delta.shape)
	syn2 +=  layer1.T.dot(layer2_delta) 
	syn1 += layer0.T.dot(layer1_delta)



# https://jeffdelaney.me/blog/useful-snippets-in-pandas/
# 19 Essential Snippets in Pandas

# https://machinelearningmastery.com/quick-and-dirty-data-analysis-with-pandas/
# quick and dirty analysis
