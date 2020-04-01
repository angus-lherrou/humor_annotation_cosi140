import os
import pickle

hashtags = set()
for file in os.listdir('/Users/stygg/Documents/201_SPR2020/COSI140/Humor/train_dir/train_data/'):
    hashtags.add('#'+''.join(file[:-4].split('_')).lower())

with open('ht.pkl', 'wb') as pkl:
    pickle.dump(hashtags, pkl)
