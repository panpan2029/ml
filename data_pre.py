import pandas as pd
import numpy as np
import os
from sklearn import linear_model
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectPercentile, f_classif, chi2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pylab import scatter, show, legend, xlabel, ylabel

# Mar 15
# use all feature
# duplicate more 1 will increase accuracy, 3 - 0.610992425714; 6 - 0.866096532334; 5 - 0.766255790867
# use selectKBest(10)
# duplicate more 1 will increase accuracy, 3 - 0.606476985822; 6 - 0.866084817245; 5 - 0.768861681006

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
    default_data = train_data[train_data[:, -1] > 0.5] # (4626, 770)
    default_data = np.c_[default_data[:, :-1], np.ones(default_data.shape[0])] # classify to one

    nondefault_data = train_data[train_data[:, -1] <= 0.5] # # (45374, 770)

    # duplicate to balance
    # for i in range(3): # 3, 0.6; 6, 0.866 (256078, 758)
    #     default_data = np.append(default_data, default_data, axis = 0)

    train8w = np.append(default_data, nondefault_data[:1000, :], axis = 0) #(82382, 770)

    univ_train8w, label_univ = univariate_feature(10, train8w)
    np.random.shuffle(univ_train8w)


    # var_train8w, label_lowVar = remove_low_variance(1, train8w)
    # print var_train8w.shape # (82382, 570) 0.8; 588 0.9; 1.0 759
    # print label_lowVar
    # np.random.shuffle(var_train8w)

    X = univ_train8w[:, :-1]
    y = univ_train8w[:, -1] 
    # X = var_train8w[:, :-1]
    # y = var_train8w[:, -1]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    print X_train.shape
    print X_test.shape

    logreg = linear_model.LogisticRegression(C = 10) #, max_iter=1000, solver = 'sag')
    logreg.fit(X_train, y_train)
    accuracy = logreg.score(X_test, y_test)

    y_pre = logreg.predict(X_test)

    print y_pre.shape[0]
    print np.count_nonzero(y_pre)
    print(accuracy)

    # visualize data, uncomment "show()" to run it
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    scatter(X[pos, 4], X[pos, 9], marker='o', c='b')
    scatter(X[neg, 4], X[neg, 9], marker='x', c='r')
    xlabel('X1')
    ylabel('X2')
    show()

    # fig, ax = plt.subplots()
    # ax.scatter(y_test, y_pre, edgecolors=(0, 0, 0))
    # ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    # ax.set_xlabel('Measured')
    # ax.set_ylabel('Predicted')
    # plt.show()

if __name__ == '__main__':
    main()
