import numpy as np
import pandas as pd
import scipy
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff

#read training data
train_set = np.load('data1000.npy')
colnames = train_set[0].decode('UTF-8').strip().split(',')
train_data = []
for elem in train_set[1:]:
    elem = elem.decode('UTF-8').split(',')
    elem = [0 if x == 'NA' else float(x) for x in elem]
    train_data.append(elem)

# train_data = pd.DataFrame(data=train_data, columns=colnames)

# data = [go.Histogram(x=[x for x in train_data['loss'] if x != 0])]
# py.iplot(data, filename='basic histogram')
print(train_data)
print(train_data.shape)