import numpy as np
import os
import matches_naive

GOLD_DIR = "gold"
KAPPA_DIR = "kappa"


def adjudicate(directory):
    if not os.path.exists(GOLD_DIR):
        os.mkdir(GOLD_DIR)
    if not os.path.exists(KAPPA_DIR):
        os.mkdir(KAPPA_DIR)

    for root, dirs, files in os.walk(directory, topdown=True):
        if len(dirs) == 0:
            print("doing it")
            annotators = []
            for name in files:
                annotators.append(str(name.split('_')[0]))
            anno_dir = os.path.join(KAPPA_DIR, '_'.join(sorted(annotators)))
            if not os.path.exists(anno_dir):
                os.mkdir(anno_dir)
            matches_naive.main(os.path.join(root, files[0]),
                               os.path.join(root, files[1]),
                               os.path.join(GOLD_DIR, '_'.join(files[0].split('_')[1:]))+'_gold.csv',
                               os.path.join(anno_dir, '_'.join(files[0].split('_')[1:]))+'_kappa.csv')


if __name__ == '__main__':
    adjudicate('test_adjudication_files')
