import numpy as np
import os
from matches_naive import num_labels
from iaa import KAPPA_DIR


def kappa(directory):
    arr = np.zeros((num_labels, num_labels))
    for filepath in os.listdir(directory):
        arr += np.loadtxt(filepath, delimiter=',')
    total = arr.sum()
    p_o = arr.trace() / total
    p_e = 0
    for i in range(len(arr)):
        p_e += np.sum(arr[i])*np.sum(arr[:, i])
    p_e /= (total**2)
    return (p_o - p_e)/(1 - p_e)


if __name__ == '__main__':
    for folder in os.listdir(KAPPA_DIR):
        print(folder, kappa(folder))
