import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def submit(model, test, fname):
    Y_test = model.predict(test.values)
    df = pd.DataFrame({'ImageId': np.arange(1, Y_test.shape[0] + 1),
                       'Label': Y_test})
    fpath = os.path.join('submissions', fname)
    df.to_csv(fpath, index=False)
    return pd.read_csv(fpath)


def load_train_dataset():
    train = pd.read_csv('dataset/train.csv')
    return train.iloc[:, 1:], train.iloc[:, 0]


def load_test_dataset():
    test = pd.read_csv('dataset/test.csv')
    return test


def drop_zero_mean(train):
    return train.loc[:, (train.describe().loc['mean'] != 0).values]


def display(train, labels, img_num, label, cmap=None):
    imgs = train[labels == label]
    img = imgs.iloc[img_num].values.reshape((28, 28))
    if cmap:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.title(label)
    plt.show()


def to_black_scale(train):
    return train.apply(lambda x: x.clip(0, 1))
