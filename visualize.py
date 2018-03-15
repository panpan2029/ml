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

data = np.load("data5_205_label.npy") #50000 x 205 + label

# x = range(50000)
# y = [i for i in data[:,data.shape[1]-1]]
label = data[:,data.shape[1]-1]
unique_val, cnt_val = np.unique(label, return_counts=True)
dict(zip(unique, counts)) 
num_zero = (label == 0).sum() # 45374 # https://stackoverflow.com/questions/28663856/how-to-count-the-occurrence-of-certain-item-in-an-ndarray-in-python/28663910
plt.plot(unique_val,cnt_val)
plt.show() 

# x = np.array([0,1,2,3])
# y = np.array([20,21,22,23])
# my_xticks = ['John','Arnold','Mavis','Matt']
# plt.xticks(x, my_xticks)
# plt.plot(x, y)
# plt.show()

