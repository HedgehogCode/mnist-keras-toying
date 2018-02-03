import numpy as np
import pandas as pd

def one_hot(lab):
    lab_int = lab.astype(int)
    res = np.zeros((len(lab_int), max(lab_int)+1))
    res[np.arange(len(lab_int)), lab_int] = 1
    return res

def classes(lab):
    return np.argmax(lab, axis=1)

def mnist_train_kaggle(path, val_split=0.9):
    data = pd.read_csv(path, delimiter=',').values
    train_n = int(len(data) * val_split)
    train_img = data[0:train_n,1:] / 255
    train_lab = one_hot(data[0:train_n,0])
    val_img = data[train_n:,1:] / 255
    val_lab = one_hot(data[train_n:,0])
    return train_img, train_lab, (val_img, val_lab)

