# final Mar 2018
# -*- coding: utf-8 -*-

from sklearn import datasets
import math
import sys
import random
import types
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scipy
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff


# >>> mb = np.array([[1,0,1,2],[1,0,3,4]])
# >>> np.where(~mb.any(axis=0))[0]
# array([1])
# https://stackoverflow.com/questions/23726026/finding-which-rows-have-all-elements-as-zeros-in-a-matrix-with-numpy
# find all zeors
# data_noid = train_data[:, 1:]
# np.where(~d_noid.any(axis=0))[0] #array([ 30,  31,  32,  34,  35, 668, 690, 691, 692])

from scipy.stats.stats import pearsonr

# print pearsonr(d_noid[:,20], d_noid[:, 25])
# # (0.10382658351877774, 7.289870861222987e-120)
# np.corrcoef(d_noid[:,20], d_noid[:,25])
# array([[1.        , 0.10382658],
#        [0.10382658, 1.        ]])

d = np.load("data1000arr.npy")
data_noid = d[:, 1:]
# data_noid=data_noid.fillna(0)

corr_stat = []
error_col = []

for i in range(0, data_noid.shape[1]-1):
	if i in [ 30,  31,  32,  34,  35, 668, 690, 691, 692]: 
		continue
	for j in range(i+1, data_noid.shape[1]):
		if j in [ 30,  31,  32,  34,  35, 668, 690, 691, 692] :
			continue
		corr = pearsonr(data_noid[:,i], data_noid[:, j])
		if corr[0] > 0.8 or corr[0] < -0.8 :
			try:
				corr_stat.append([i,j, corr[0]])
			except:
				print i, j, "warning"
			print 'proceeding...'

		# try:
		# 	corr = pearsonr(data_noid[:,i], data_noid[:, j])
		# # print corr
		# 	if corr[0] > 0.8 or corr[0] < -0.8 :
		# 		# print corr[0]
		# 		corr_stat.append([i,j,corr[0]])
		# except ValueError:
		# 	error_col.append([i,j])
		# 	continue
		# # corr_cols.append([i,j])
# print corr_stat
np.save("corr_cols.npy", corr_stat)
sys.exit(0)

# 111111
trainfile = "ecs171train.npy"
#read training data
train_set = np.load(trainfile)
# colnames = train_set[0].decode('UTF-8').strip().split(',')
train_data = []
for elem in train_set[1:]:
	elem = elem.decode('UTF-8').split(',')
	elem = [0 if x == 'NA' else float(x) for x in elem]
	train_data.append(elem)

train_data = np.array(train_data) # id + 769 features + loss
print(train_data.shape)
np.save("data1000arr.npy", train_data)






# Notice that the numerator of this fraction is identical to the above definition of covariance, since mean and expectation can be used interchangeably. Dividing the covariance between two variables by the product of standard deviations ensures that correlation will always fall between -1 and 1. This makes interpreting the correlation coefficient much easier.

# correlation 
# https://www.datascience.com/blog/introduction-to-correlation-learn-data-science-tutorials