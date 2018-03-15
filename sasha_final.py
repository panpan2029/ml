import pandas as pd
import numpy as np
import os
from sklearn import linear_model


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
    if os.path.exists('train.df'):
        train_df = pd.read_pickle('train.df')
    else:
        train_df = read_data("ecs171train.npy")
        train_df.to_pickle('train.df')
    # if os.path.exists('test.df'):
    #     test_df = pd.read_pickle('test.df')
    # else:
    #     test_df = read_data("ecs171test.npy")
    #     test_df.to_pickle('test.df')
    main = train_df
    nunique = main.apply(pd.Series.nunique)
    drop_cols = nunique[nunique == 1].index
    main.drop(drop_cols, axis=1, inplace=True)

    msk = np.random.rand(len(main)) < 0.8
    train_df = main[msk]
    test_df = main[~msk]
    train_x = train_df.iloc[:, :-1]
    train_y = train_df.iloc[:, -1]
    test_x = test_df.iloc[:, :-1]
    test_y = test_df.iloc[:, -1]
    train_y = train_y.apply(lambda x: 0 if x == 0 else 1)
    test_y = test_y.apply(lambda x: 0 if x == 0 else 1)

    logreg = linear_model.LogisticRegression()
    logreg.fit(train_x, train_y)
    accuracy = logreg.score(test_x, test_y)
    print(accuracy)


if __name__ == '__main__':
    main()
