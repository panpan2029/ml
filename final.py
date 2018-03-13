# final Mar 2018
# -*- coding: utf-8 -*-
from __future__ import division
from sklearn import datasets

import math
import sys
import random
import types
import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns



trainfile = "ecs171train.npy"
testfile = "ecs171test.npy"
csvfile = "train_data_nomissing.csv"
csvtest1 = "test.csv"
csvtest2 = "test_missing.csv"
csvfile_missing = "train_data_missing.csv"

data = np.load(testfile) 
# print(data[0])
Xstr = []
Xstr_missing = []
# for i in range(500):
# 	xs = data[i].split(',')
# 	Xstr.append(xs)
# k = 0
# for d in data:
# 	xs = [float(i) for i in d.split(',')]
	# xs = d.split(',')
	# Xstr.append(xs)
	# print(d)
	# print(xs)
	# k += 1
	# if k > 500 :
	# 	break

for d in data:
	try:
		xs = [float(i) for i in d.split(',')]
		Xstr.append(xs)
	except ValueError:
		xs = d.split(',')
		Xstr_missing.append(xs)
		continue;

Xstr = np.array(Xstr) #(55471, 770)
Xstr_missing = np.array(Xstr_missing) #(55471, 770)
print("data : {0}".format(Xstr.shape))
print("data with missing features : {0}".format(Xstr_missing.shape))

df = pd.DataFrame(Xstr)
df.to_csv(csvtest1)
# df.to_csv(csvfile)

df = pd.DataFrame(Xstr_missing)
df.to_csv(csvtest2)

# print(Xstr.shape)
# np.savetxt(csvfile, Xstr, delimiter=',')

# data_str = np.arange(data.shape[0])
# data_str = np.chararray((data.shape[0], ))

# print(data_str.shape)
# np.savetxt('test.out', data,delimiter=',')
# np.random.rand(2,3) * 2 -1

# for i in range(data.shape[0]):
# 	data_str[i] = data[i]
# print(data_str)
# sometime 70 features, sometime 770 features