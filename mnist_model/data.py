import numpy as np
import pandas as pd

def one_hot(lab):
    lab_int = lab.astype(int)
    res = np.zeros((len(lab_int), max(lab_int)+1))
    res[np.arange(len(lab_int)), lab_int] = 1
    return res

def classes(lab):
    return np.argmax(lab, axis=1)

def normalize(data):
    return (data - np.mean(data, axis=1)[:,None]) / np.std(data, axis=1)[:,None]

def mnist_train_kaggle(path, val_split=0.9):
    data = pd.read_csv(path, delimiter=',').values
    train_n = int(len(data) * val_split)
    train_img = normalize(data[0:train_n,1:])
    train_lab = one_hot(data[0:train_n,0])
    val_img = normalize(data[train_n:,1:])
    val_lab = one_hot(data[train_n:,0])
    return train_img, train_lab, (val_img, val_lab)

def mnist_test_kaggle(path):
    return pd.read_csv(path, delimiter=',').values / 255

def save_submission_kaggle(path, labs):
    values = np.concatenate((np.arange(1,len(labs)+1)[...,None], labs[...,None]), axis=1)
    data_frame = pd.DataFrame(data=values, dtype=np.int32, columns=['ImageId','Label'])
    data_frame.to_csv(path_or_buf=path, index=False)
