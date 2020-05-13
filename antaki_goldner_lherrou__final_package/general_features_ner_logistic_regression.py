#Joseph Antaki
import os, re
from collections import defaultdict
import spacy
import numpy as np
from nltk import word_tokenize, pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from profanity_check import predict_prob

humor_labels = ['ABSD', 'IRNY', 'ISLT', 'OBSV', 'VLGR', 'WPLY', 'OTHR', 'NHMR']

def process(path='gold/'):
    d = {}
    url_pattern = re.compile('https://t.co/[A-Za-z0-9]+')
    midnight_pattern = re.compile('@midnight')
    for root, dir, files in os.walk(path):
        for name in files:
            tweets = []
            with open(os.path.join(root, name)) as csv:
                hashtag = '#' + name[:name.find('.')].replace('_', '')
                hashtag_pattern = re.compile(hashtag, re.IGNORECASE)
                i = iter(csv)
                next(i) # skip the first line of labels
                while True:
                    try:
                        line = next(i).strip()
                        if len(line) != 0:
                            parts = []
                            first_comma = line.find(',')
                            last_comma = line.rfind(',')
                            parts.append(line[:first_comma])
                            parts.append(line[first_comma + 1 : last_comma])
                            parts.append(line[last_comma + 1 :])
                            # Now parts = [id, tweet, humor]

                            parts[1] = re.sub(hashtag_pattern, '', parts[1])
                            parts[1] = re.sub(url_pattern, '', parts[1])
                            parts[1] = re.sub(midnight_pattern, '', parts[1])

                            if len(parts[1].strip()) == 0:
                                continue

                            # Split humor into an n-hot representation
                            parts[2] = present(parts[2].split('_'))

                            tweets.append(parts)
                    except StopIteration:
                        break

                d[hashtag] = tweets

    return d

def train_test(d):
    samples = []
    labels = []
    unwanted_named_entites = set(['DATE','TIME','PERCENT','MONEY','QUANTITY', 'ORDINAL', 'CARDINAL'])
    nlp = spacy.load('en_core_web_sm')

    for hashtag in d.values():
        for id, tweet, humor in hashtag:
            features = []
            ents_in_tweet = 0
            doc = nlp(tweet)
            for ent in doc.ents:
                if ent.label_ not in unwanted_named_entites:
                    ents_in_tweet += 1

            features.extend(predict_prob([tweet])) # proability of offensiveness via profanity
            features.append(ents_in_tweet) # named entity count
            features.append(len(tweet)) # length of tweet in characters

            tokenized = word_tokenize(tweet)
            if len(tokenized) == 0:
                print(id)
            features.append(len(tokenized)) # word count
            features.append(sum(map(len, tokenized)) / len(tokenized)) # average word length

            tagged = pos_tag(tokenized, tagset='universal')
            tag_freq = defaultdict(int)
            for (word, tag) in tagged:
                tag_freq[tag] += 1

            features.append(tag_freq['NOUN']) # noun count
            features.append(tag_freq['VERB']) # verb count
            features.append(tag_freq['ADJ']) # adj count
            features.append(tag_freq['ADV']) # adv count
            features.append(tag_freq['.'])   # punctuation count
            features.append(tag_freq['NOUN'] / len(tokenized))  # noun count to total word count
            features.append(tag_freq['VERB'] / len(tokenized))  # verb count to total word count
            features.append(tag_freq['ADJ'] / len(tokenized))  # adj count to total word count
            features.append(tag_freq['ADV'] / len(tokenized))  # adv count to total word count

            synset_counts = list(map(len, map(wn.synsets, tokenized)))
            avg_synset_count = sum(synset_counts) / len(tokenized)
            features.append(avg_synset_count) # average synset count per word
            features.append(max(synset_counts)) # max synset count a word has
            features.append(max(synset_counts) - avg_synset_count) # difference between greatest and avg synset count

            features.extend(get_intensities(tagged)) # multiple intensity related

            samples.append(features)
            labels.append(humor)

    samples = preprocessing.scale(samples)

    samples = np.array(samples)
    labels = np.array(labels)

    x_train, x_test, y_train, y_test = train_test_split(samples, labels, test_size=0.33)

    lr = LogisticRegression(multi_class='multinomial', max_iter=500)
    # Logistic Regression doesn't handle multiple classification natively,
    # so MultiOutputClassifier can brute-force it into acting like it does
    moc = MultiOutputClassifier(lr)
    moc = moc.fit(x_train, y_train)
    print("Accuracy:", moc.score(x_test, y_test))

    y_pred = moc.predict(x_test)
    print(classification_report(y_test, y_pred, target_names=humor_labels))

def get_intensities(tagged):
    code_to_list = {'a': [], 'r':[]}
    final = []

    for (word, pos) in tagged:
        if pos == 'ADV':
            code = 'r'
        elif pos == 'ADJ':
            code = 'a'
        else:
            #If the word is not an adjective or adverb, don't score it
            continue
        l = lookup(word, code)
        if l is not None: # if the word has a score in sentiwordnet
            code_to_list[code].append(l)

    for scores in code_to_list.values():
        total = sum(scores)
        if len(scores) == 0:
            avg = 0
            highest = 0
        else:
            avg = total/len(scores)
            highest = max(scores)

        final.append(total) # sum of all scores
        final.append(avg) # avg score
        final.append(highest) # max score
        final.append(highest - avg) # gap
    return final

def lookup(word, pos_code):
    lm = WordNetLemmatizer()
    s = list(swn.senti_synsets(lm.lemmatize(word), pos_code))
    if len(s) != 0:
        # As a measure of intensity, take the highest score between the word's
        # positive and negative scores
        return max([s[0].pos_score(), s[0].neg_score()])
    else:
        return None

'''
Thanks to Eli Goldner for this function
'''
def present(guesses):
    results = np.zeros(len(humor_labels))
    for idx in range(len(humor_labels)):
        if humor_labels[idx] in guesses:
            results[idx] += 1
    return results

if __name__ == "__main__":
    data = process()
    train_test(data)