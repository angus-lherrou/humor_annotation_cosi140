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

totals = {'ANC-spoken-freqs.txt' : 3862172, 'ANC-written-freqs.txt': 18302813}
humor_labels = ['ABSD', 'IRNY', 'ISLT', 'OBSV', 'VLGR', 'WPLY', 'OTHR', 'NHMR']
penn_to_universal = {'EX':'PRON', 'MD':'VERB', 'WDT':'DET', 'VBD':'VERB', 'VBN':'VERB', 'RBS':'ADV',
                     'VBZ':'VERB', 'JJR':'ADJ', 'MD|VB':'VERB', 'TO':'PART', 'RB':'ADV', 'DT':'DET',
                    'RBR':'ADV', 'SYM':'SYM', 'NNS|VBZ':'X', 'IN':'ADP', 'VB':'VERB', 'NNS':'NOUN',
                     'CC':'CCONJ', 'NN|CD':'NUM', 'FW':'X', 'PRP$':'DET', 'WP$':'DET',
                     'NN':'NOUN', 'VBP':'VERB', 'POS':'PART', 'UH':'INTJ',
                     'JJ':'ADJ', 'VBG|NN':'X', 'NNP':'NOUN', 'RP':'ADP',
                     'JJS':'ADJ', 'VBG':'VERB', 'NNPS':'NOUN', 'UNC':'X',
                      'PRP':'PRON', 'PDT':'DET', 'WRB':'ADV', 'WP':'PRON',
                      'NN|JJ':'NOUN'}

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

def train_test(d, spoken=None, written=None):
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

            #tweet_spoken_freqs = freq_features(tagged, spoken)
            #tweet_written_freqs = freq_features(tagged, written)
            #features.extend(tweet_spoken_freqs) # multiple
            #features.extend(tweet_written_freqs) # multiple
            #features.append(tweet_written_freqs[0] - tweet_spoken_freqs[0]) # difference between averages

            samples.append(features)
            labels.append(humor)

    samples = preprocessing.scale(samples)

    #Convert to ndarray since the fellas at sklearn don't consistently treat lists as "array-like"
    samples = np.array(samples)
    labels = np.array(labels)

    x_train, x_test, y_train, y_test = train_test_split(samples, labels, test_size=0.33)

    lr = LogisticRegression(multi_class='multinomial', max_iter=500)
    moc = MultiOutputClassifier(lr)
    moc = moc.fit(x_train, y_train)
    #lr = lr.fit(x_train, y_train)
    print("Accuracy:", moc.score(x_test, y_test))

    y_pred = moc.predict(x_test)
    print(classification_report(y_test, y_pred, target_names=humor_labels))

def get_frequencies(source):
    freqs = defaultdict(lambda: defaultdict(int))
    with open(source) as file:
        for line in file:
            line = line.split('\t')
            inner = defaultdict(int)
            inner[penn_to_universal[line[2]]] = float(line[3]) / totals[source]
            freqs[line[1]] = inner

    return freqs

def freq_features(tagged, freqs):
    lm = WordNetLemmatizer()
    frequencies = []
    for i in range(len(tagged)):
        frequencies.append(lm.lemmatize(tagged[i][0]))
        frequencies[i] = freqs[frequencies[i]][tagged[i][1]]

    final = []
    final.append(sum(frequencies) / len(frequencies)) # average over tweet
    final.append(min(frequencies)) # most uncommon
    final.append(max(frequencies) - min(frequencies)) # difference between most and least common
    return final

def get_intensities(tagged):
    code_to_list = {'a': [], 'r':[]}
    final = []

    for (word, pos) in tagged:
        if pos == 'ADV':
            code = 'r'
        elif pos == 'ADJ':
            code = 'a'
        else:
            continue
        l = lookup(word, code)
        if l is not None:
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
    #spoken_freqs = get_frequencies('ANC-spoken-freqs.txt')
    #written_freqs = get_frequencies('ANC-written-freqs.txt')
    train_test(data)