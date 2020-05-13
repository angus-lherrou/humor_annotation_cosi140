import nltk
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neural_network import MLPClassifier
from operator import itemgetter
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import sentiwordnet as swn
import pandas as pd
import glob
import numpy as np
import os
import re
from pywsd import simple_lesk, disambiguate
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


def sense_relate(sentence, context='all', window=1):
    print("relating...")
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
            print("counted")
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
    labels = ['train']*train_count + ['test']*test_count
    random.shuffle(labels)
    df['type'] = labels
    for index, row in df.iterrows():
        if type(row['tweet_text']) != str:
            df.drop(index, inplace=True)
    return df


def featurize(csv_path, feat_vocab, stemmer, skip=1):
    cols = ['_type_', '_label_']
    cols.extend(list(feat_vocab))
    cols.extend(['sense_changes', 'length'])

    data_frame = read_golds(csv_path)

    row_count = data_frame.shape[0]
    print(row_count)
    feat_data_frame = pd.DataFrame(index=np.arange(row_count), columns=cols)
    feat_data_frame.fillna(0, inplace=True)  # inplace: mutable
    for index, row in data_frame.iterrows():
        if index % skip == 0:
            print(index, f'of {data_frame.shape[0]}')
            feat_data_frame.loc[index, '_type_'] = row['type']
            if 'WPLY' in set(row['tweet_classifications'].split('_')):
                feat_data_frame.loc[index, '_label_'] = 'WPLY'
            else:
                feat_data_frame.loc[index, '_label_'] = 'NOT_WPLY'
            for token in stem_tokens(stemmer, row['tweet_text']):
                if token in feat_vocab:
                    feat_data_frame.loc[index, token] += 1
                    feat_data_frame.loc[index, 'length'] += 1
            feat_data_frame.loc[index, 'sense_changes'] = sense_changes(' '.join(tokenize(row['tweet_text'])))

    return feat_data_frame


def vectorize(df, mlb=None):
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
    if mlb:
        combined_labels = [label.split('_') for label in df._label_.values]
        labels = mlb.transform(combined_labels)
    else:
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


def classify(feat_csv, mlb: MultiLabelBinarizer = None):
    # Moved the csv loading and splitting
    # to here for easier control
    df = pd.read_csv(feat_csv, encoding='latin1')
    #init_train, init_test = train_test_split(df, test_size=0.2)
    init_train, init_test = custom_split(df)
    print(init_train.shape, init_test.shape)

    if mlb:
        X_train, y_train = vectorize(init_train, mlb)
        X_test, y_test = vectorize(init_test, mlb)
        model = MLPClassifier(solver='lbfgs',
                              verbose=0,
                              max_iter=500)
        model = train_model(X_train, y_train, model)
        accuracy, report, predictions = test_model(X_test, y_test, model, mlb.classes_)
    else:
        X_train, y_train = vectorize(init_train)
        X_test, y_test = vectorize(init_test)
        model = LogisticRegression(multi_class='multinomial',
                                   penalty='l2',
                                   solver='lbfgs',
                                   max_iter=500,
                                   verbose=1)
        model = train_model(X_train, y_train, model)
        accuracy, report, predictions = test_model(X_test, y_test, model)

    print(report)
    return accuracy, report, predictions


def cohen_kappa(y_gold, y_pred, labels):
    print("""Cohen's Kappa scores:""")
    for index, label in enumerate(labels):
        arr = np.zeros((2, 2))
        for gold_sample in y_gold:
            for pred_sample in y_pred:
                arr[gold_sample[index], pred_sample[index]] += 1
        total = arr.sum()
        p_o = arr.trace() / total
        p_e = 0
        for i in range(len(arr)):
            p_e += np.sum(arr[i]) * np.sum(arr[:, i])
        p_e /= (total ** 2)
        print(f"    {label}    {(p_o - p_e) / (1 - p_e)}")


if __name__ == '__main__':
    ps = PorterStemmer()
    feat_vocab = extract_feat_vocab('gold', ps)
    print(len(feat_vocab))
    selected_feat_vocab = select_features(feat_vocab)
    feat_data_frame = featurize('gold', selected_feat_vocab, ps)
    featfile = os.path.join(os.path.curdir, "wordplay_features.csv")
    feat_data_frame.to_csv(featfile, encoding='latin1', index=False)
    accuracy, report, predictions = classify('wordplay_features.csv')
