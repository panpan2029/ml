import pandas as pd
import numpy as np
import os

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


def main():

	# df = pd.DataFrame(data=data_list, columns=column_name)
	# [id, 769 features, loss]
	if os.path.exists('train.df'):
		train_df = pd.read_pickle('train.df')
	else:
		train_df = read_data("ecs171train.npy")
		train_df.to_pickle('train.df')

	main = train_df # (50000, 771), id+features+loss
	nunique = main.apply(pd.Series.nunique) # column : unique value cnt
	cols_uniqueValue = nunique[nunique == 1].index # 11 : Index([u'f33', u'f34', u'f35', u'f37', u'f38', u'f678', u'f700', u'f701', u'f702', u'f736', u'f764'],
	main.drop(cols_uniqueValue, axis=1, inplace=True)
	
	print "Correlation Matrix......"
	if os.path.exists('corr.df'):
		corrMatrix = pd.read_pickle('corr.df')
	else:
		corrMatrix = main.corr().abs()
		corrMatrix.to_pickle('corr.df')
	corrColum = corrMatrix.unstack()
	# corrColum = corrMatrix.unstack().reset_index()
	# corrColum.columns = ['c1','c2','corr_value']

	print "List highly correlated columns Rank......" # only for view
	import itertools
	if os.path.exists('corrPairs.df'):
		corrPairs = pd.read_pickle('corrPairs.df')
	else:
		corrColum = corrMatrix.unstack()
		corrPairs = pd.DataFrame([[(i,j),corrMatrix.loc[i,j]] for i,j in list(itertools.combinations(corrMatrix, 2))],columns=['pairs','corr']) # (u'f745', u'f746') 0.127421944815
		corrPairs.to_pickle('corrPairs.df')

	# corrPairs_sort = corrPairs.sort_values(by = 'corr', ascending = False) # shape (288420, 2)
	# print corrPairs_sort.loc[:100, :]

	print "Pairs_to_drop..0.9...."	
	pairs_to_drop = set()
	labels_to_drop = set()
	cols = corrMatrix.columns
	for i in corrMatrix.columns[1:-1]: # no id  , i column name, j index
		for j in range(1, corrMatrix.columns.get_loc(i)):
			iid = corrMatrix.columns.get_loc(i)
			jname = cols[j]
			# print jname
			if corrMatrix[i][jname]>=0.9 :
				pairs_to_drop.add((cols[iid], cols[j]))
				labels_to_drop.add(cols[j])

	main.drop(labels = labels_to_drop, axis=1, inplace=True) # shape (50000, 286) after removing corr > 0.9
	
	# print "Label Frequency......"
	# labels_freq = np.zeros(101)
	# print main['loss']
	# for i in main['loss']:
	# 	labels_freq[int(i)] += 1
 # [4.5374e+04 5.4400e+02 6.3100e+02 5.1800e+02 4.8700e+02 3.1700e+02
 # 2.4600e+02 2.6900e+02 2.3100e+02 1.6400e+02 1.5800e+02 1.0600e+02
 # 1.0100e+02 9.0000e+01 7.0000e+01 5.9000e+01 5.5000e+01 5.8000e+01
 # 5.0000e+01 4.3000e+01 3.6000e+01 2.8000e+01 3.8000e+01 2.2000e+01
 # 2.3000e+01 2.0000e+01 1.3000e+01 2.3000e+01 1.6000e+01 1.5000e+01
 # 5.0000e+00 1.2000e+01 4.0000e+00 1.2000e+01 2.0000e+00 3.0000e+00
 # 3.0000e+00 1.2000e+01 4.0000e+00 1.0000e+01 5.0000e+00 1.0000e+01
 # 1.0000e+00 9.0000e+00 4.0000e+00 5.0000e+00 3.0000e+00 4.0000e+00
 # 4.0000e+00 6.0000e+00 4.0000e+00 2.0000e+00 1.0000e+00 1.0000e+00
 # 8.0000e+00 1.0000e+00 1.0000e+00 1.0000e+00 1.0000e+00 3.0000e+00
 # 2.0000e+00 2.0000e+00 1.0000e+00 0.0000e+00 4.0000e+00 2.0000e+00
 # 0.0000e+00 0.0000e+00 0.0000e+00 4.0000e+00 1.0000e+00 0.0000e+00
 # 1.0000e+00 0.0000e+00 0.0000e+00 0.0000e+00 0.0000e+00 1.0000e+00
 # 2.0000e+00 5.0000e+00 1.0000e+00 0.0000e+00 0.0000e+00 0.0000e+00
 # 1.0000e+00 0.0000e+00 0.0000e+00 0.0000e+00 0.0000e+00 2.0000e+00
 # 2.0000e+00 0.0000e+00 0.0000e+00 0.0000e+00 3.0000e+00 1.0000e+00
 # 0.0000e+00 0.0000e+00 0.0000e+00 1.0000e+00 1.8000e+01]

	print "Balance the data......"
	# print main.columns
	# default50 = main[ main['loss'] > 50.0]  # 73
	# print default50.shape
	# default40 = main[ (main['loss'] > 40.0) & (main['loss'] < 50.0)] # 46
	# print default40.shape 
	# default30 = main[ (main['loss'] > 30.0 )& (main['loss'] < 40.0)] # (62, 286)
	# print default30.shape 
	# default20 = main[( main['loss'] > 20.0) & (main['loss'] < 30.0)] # (198, 286)
	# print default20.shape 
	# default10 = main[ (main['loss'] > 10.0) & (main['loss'] < 30.0)] # (866, 286)
	# print default10.shape 

	# non_default = main[ main['loss'] <= 0.9] # (45374, 286)
	# print non_default.shape

	# print label_to_drop
 #    cols = df.columns
 #    for i in range(0, df.shape[1]):
 #        for j in range(0, i+1):
 #            pairs_to_drop.add((cols[i], cols[j]))
 #    return pairs_to_drop


 # def corrank(X):
 #        import itertools
 #        df = pd.DataFrame([[(i,j),X.corr().loc[i,j]] for i,j in list(itertools.combinations(X.corr(), 2))],columns=['pairs','corr'])    
 #        print(df.sort_values(by='corr',ascending=False))

  # corrank(X)


	# main.drop(drop_cols, axis=1, inplace=True)


	return 
	# seperate default and non default data, np.array
	train_data = np.array(train_df)[:, 1: ] # remove id
	# draw_density(train_data)
	default_data = train_data[train_data[:, -1] > 0.5] # (4626, 770)
	# default_data = np.c_[default_data[:, :-1], np.ones(default_data.shape[0])] # classify to one
	nondefault_data = train_data[train_data[:, -1] <= 0.5] # # (45374, 770)
 
   

if __name__ == '__main__':
	main()
