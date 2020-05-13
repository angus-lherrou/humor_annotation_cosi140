import nltk
import random
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.neural_network import MLPClassifier
from operator import itemgetter
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
import glob
import numpy as np
import os
import json
from pywsd.similarity import max_similarity


# Filtering out irrelevant content
stopwords = set(word.lower() for word in nltk.corpus.stopwords.words('english'))
punctuation = {'\'', '\"', ',', '.', '-',
               '--', '?', '!', '\'\'', '\\',
               ':', ';', '`', '``', '/', '\'s', '(',
               ')', '{', '}', '[', ']', '#'}
pronouns = {'i', 'he', 'she', 'they', 'you', 'me', 'mine', 'your',
            'him', 'his', 'her', 'hers', 'them', 'theirs'}
notcontent = stopwords | punctuation | pronouns

LABELS = [['ABSD', 'IRNY', 'ISLT', 'NHMR', 'OBSV', 'OTHR', 'VLGR', 'WPLY']]

with open('train_test.json', 'r') as fp:
    SPLIT = json.load(fp)


def sense_relate(sentence, context='all', window=1):
    tokens = sentence.split()
    sense_list = []
    if context == 'all':
        for word in tokens:
            sense_list.append(max_similarity(sentence, word, 'wup'))
    elif context == 'window':
        for x, word in enumerate(tokens):
            if x == 0:
                sense_list.append(max_similarity(' '.join(tokens[:x + window + 1]), word, 'wup'))
            elif x == len(tokens)-1:
                sense_list.append(max_similarity(' '.join(tokens[x - window:]), word, 'wup'))
            else:
                sense_list.append(max_similarity(' '.join(tokens[x - window:x + window + 1]), word, 'wup'))
    else:
        raise AttributeError("context should be 'all' or 'window'")
    return sense_list


def sense_changes(sentence):
    senses_1 = sense_relate(sentence, context='all')
    senses_2 = sense_relate(sentence, context='window', window=1)
    counter = 0
    for sense_1, sense_2 in zip(senses_1, senses_2):
        if sense_1 and sense_2 and sense_1._name == sense_2._name:
            counter += 1
    return counter


def stem_tokens(stemmer, text):
    initial = tokenize(text)
    return [stemmer.stem(token) for token in initial]


def tokenize(text):
    intermediate = word_tokenize(text)
    final = []
    tagged = False
    for item in intermediate:
        if not tagged:
            final.append(item)
        tagged = item == '#' or item == '@'
    return final


def extract_feat_vocab(csv_path, stemmer):
    data_frame = read_golds(csv_path)
    feat_vocab = dict()
    for index, row in data_frame[data_frame['type'] == 'train'].iterrows():
        for token in stem_tokens(stemmer, row['tweet_text']):
            feat_vocab[token] = feat_vocab.get(token, 0) + 1
    return feat_vocab


def select_features(feat_vocab):
    sorted_feat_vocab = sorted(feat_vocab.items(), key=itemgetter(1), reverse=True)
    feat_dict = dict(sorted_feat_vocab)
    return set(feat_dict.keys())


def read_golds(csv_path):
    all_files = glob.glob(csv_path + "/*.csv")
    li = []
    for filename in all_files:
        df = pd.read_csv(filename)
        li.append(df)
    df = pd.concat(li, ignore_index=True)
    train_count = int(df.shape[0] * 0.8)
    test_count = df.shape[0] - train_count

    global SPLIT

    if len(SPLIT) != df.shape[0]:
        SPLIT = ['train']*train_count + ['test']*test_count
        random.shuffle(SPLIT)
    df['type'] = SPLIT
    for index, row in df.iterrows():
        if type(row['tweet_text']) != str:
            df.drop(index, inplace=True)
    return df


def featurize(csv_path, feat_vocab, stemmer, skip=1):
    cols = ['_type_', '_label_']
    cols.extend(list(feat_vocab))
    cols.extend(['_sense_changes_', '_length_', '_sense_changes_by_length_'])

    data_frame = read_golds(csv_path)

    row_count = data_frame.shape[0]
    print(row_count)
    feat_data_frame = pd.DataFrame(index=np.arange(row_count), columns=cols)
    feat_data_frame.fillna(0, inplace=True)  # inplace: mutable
    for index, row in data_frame.iterrows():
        if index % skip == 0:
            if index % (data_frame.shape[0]//25) == 0:
                print(index, f'of {data_frame.shape[0]}')
            feat_data_frame.loc[index, '_type_'] = row['type']
            if 'WPLY' in set(row['tweet_classifications'].split('_')):
                feat_data_frame.loc[index, '_label_'] = 1
            else:
                feat_data_frame.loc[index, '_label_'] = 0
            length = 0
            for token in stem_tokens(stemmer, row['tweet_text']):
                if token in feat_vocab:
                    feat_data_frame.loc[index, token] += 1
                length += 1
            feat_data_frame.loc[index, '_length_'] = length
            sense_change_count = sense_changes(' '.join(tokenize(row['tweet_text'])))
            feat_data_frame.loc[index, '_sense_changes_'] = sense_change_count
            feat_data_frame.loc[index, '_sense_changes_by_length_'] = sense_change_count / (length if length else 1)

    return feat_data_frame


def vectorize(df):
    df = df.fillna(0)
    for index, row in df.iterrows():
        if row['_type_'] == '0':
            df.drop(index, inplace=True)
    data = list()
    for index, row in df.iterrows():
        datum = dict()
        datum['bias'] = 1
        for col in df.columns:
            if not (col == "_type_" or col == "_label_" or col == 'index'):
                datum[col] = row[col]
        data.append(datum)
    vec = DictVectorizer()
    data = vec.fit_transform(data).toarray()
    print('data.shape:', data.shape)
    labels = df._label_.values
    print('labels.shape:', labels.shape)
    return data, labels


def train_model(X_train, y_train, model):
    model.fit(X_train, y_train)
    return model


def test_model(X_test, y_test, model, labels=None):
    predictions = model.predict(X_test)
    if labels:
        report = classification_report(predictions, y_test, target_names=labels)
    else:
        report = classification_report(predictions, y_test)
    accuracy = accuracy_score(predictions, y_test)
    return accuracy, report, predictions


def custom_split(df: pd.DataFrame):
    train = df[df._type_=='train']
    test = df[df._type_=='test']
    return train, test


def classify(feat_csv, labels):
    df = pd.read_csv(feat_csv, encoding='latin1')
    init_train, init_test = custom_split(df)
    print(init_train.shape, init_test.shape)

    X_train, y_train = vectorize(init_train)
    X_test, y_test = vectorize(init_test)
    model = MLPClassifier(solver='lbfgs',
                          verbose=0,
                          max_iter=500)
    model = train_model(X_train, y_train, model)
    accuracy, report, predictions = test_model(X_test, y_test, model, labels)

    print(report)
    print("Cohen's kappa score:", cohen_kappa_score(y_test, predictions))
    return accuracy, report, predictions


def extract_support_from_gold(feat_csv):
    df = pd.read_csv(feat_csv, encoding='latin1')
    init_train, init_test = custom_split(df)
    return init_test[init_test._label_==1].count(), init_test


if __name__ == '__main__':
    ps = PorterStemmer()
    feat_vocab = extract_feat_vocab('gold', ps)
    print(len(feat_vocab))
    selected_feat_vocab = select_features(feat_vocab)
    feat_data_frame = featurize('gold', selected_feat_vocab, ps)
    featfile = os.path.join(os.path.curdir, "wordplay_features_word_sense.csv")
    feat_data_frame.to_csv(featfile, encoding='latin1', index=False)
    accuracy, report, predictions = classify('wordplay_features_word_sense.csv', labels=['NOT_WPLY', 'WPLY'])
