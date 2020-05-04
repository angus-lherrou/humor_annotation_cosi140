# !!CONVENTION!! - For the numpy array/outgoing csv for Cohen's Kappa
# the rows will represent annotator 1, and the columns will represent annotator 2
import os
import xml.etree.ElementTree as ET
import csv
import numpy as np
from collections import defaultdict
from itertools import chain, combinations


# lifted from https://docs.python.org/3/library/itertools.html#itertools-recipes
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

categories = {"IRNY", "ISLT", "OBSV", "WPLY", "VLGR", "ABSD", "OTHR", "NHMR"}
# labels = [set(elem) for elem in powerset(categories) if elem]
labels = [set(elem) for elem in powerset(categories)]
num_labels = len(labels)


def diff(l1, l2): 
    l_dif = []
    for elem in l1 + l2:
        if elem not in l1:
            l_dif.append("second list", l2.index(elem), elem)
        elif elem not in l2:
            l_dif.append("first list", l1.index(elem), elem)
    return l_dif 


def get_tweet_dict(xmlf1, xmlf2):
    d = defaultdict(list)
    # create element tree object 
    tree_1 = ET.parse(xmlf1)
    tree_2 = ET.parse(xmlf2)
    # get root element 
    root_1 = tree_1.getroot()
    root_2 = tree_2.getroot()
    # iterate news items
    raw_tweets_1 = root_1.find("TEXT").text
    raw_tweets_2 = root_2.find("TEXT").text
    if raw_tweets_1 == raw_tweets_2:
        for line in raw_tweets_1.split('\n'):
            temp = line.split()
            # guard here since there's an empty line at the
            # end for some reason
            if temp:
                # index by tweet number and trim the
                # tweet number and rank number from the entry
                d[temp[0]] = temp[1:-1]
        return d
    else:
        print("Error: Tweets Mismatch")
        tweets_1 = [line for line in raw_tweets_1.split('\n')]
        tweets_2 = [line for line in raw_tweets_2.split('\n')]
        for elem in diff(tweets_1, tweets_2):
            print(elem)


def parseXML(xmlfile):
    d_tags = defaultdict(set)
    d_texts = defaultdict(set)
    d_merged = defaultdict(list)
    # create element tree object 
    tree = ET.parse(xmlfile) 
    # get root element 
    root = tree.getroot() 
    # iterate news items
    for child in root.iter():
        # Checking for humor tag means it's a tagged tweet
        # (Not_Humor being one of the humor subtags)
        if child.tag == 'Humor':
            # Guard here bc sometimes tweetID was missing in the tag
            if 'fromID' in child.attrib:
                tweet = child.attrib['fromID']
                # guard boolean here bc people sometimes tagged tags
                #d[tweet] = [set(), set()]
                istweet = ''.join(char for char in tweet if char.isalpha()) == 'tweet'
                if 'toID' in child.attrib and istweet:
                    # for lack of an easier way to get tags get the 'OBSV' or 'WPLY'
                    # (observational and wordplay) from pointers to tags like 'OBSV45' etc
                    tag = ''.join(char for char in child.attrib['toID'] if char.isalpha())
                    d_tags[tweet].add(tag)
                if 'fromText' in child.attrib and istweet:
                    d_texts[tweet].add(child.attrib['fromText'])
    for key in d_tags.keys():
        if d_tags[key]:
            d_merged[key] = [d_texts[key], d_tags[key]]
        else:
            d_merged[key] = [d_texts[key], {}]
    return d_merged


def resolve(f1, f2, outfile):
    # probably a way to generalize this into one forloop
    # without all the variable names
    ck_mtx = np.zeros((num_labels, num_labels))
    tweet_dict = get_tweet_dict(f1, f2)
    tagged_tweets_1 = parseXML(f1)
    tagged_tweets_2 = parseXML(f2)
    discrepancies = defaultdict(dict)
    results = defaultdict(list)
    for key in tagged_tweets_1.keys():
        # defaults so no unmentioned var errors
        an1_label = set()
        an2_label = set()
        if tagged_tweets_1[key] == tagged_tweets_2[key]:
            textid = list(tagged_tweets_1[key][0])[0]
            # add text and tags
            results[key] = [' '.join(tweet_dict[textid]), tagged_tweets_1[key][1]]
            an1_label = tagged_tweets_1[key][1]
            an2_label = tagged_tweets_2[key][1]
        else:
            if tagged_tweets_1[key] and tagged_tweets_2[key]:
                discrepancies[key][f1] = tagged_tweets_1[key]
                discrepancies[key][f2] = tagged_tweets_2[key]
                an1_label = tagged_tweets_1[key][1]
                an2_label = tagged_tweets_2[key][1]
            elif not tagged_tweets_1[key]:
                discrepancies[key][f1] = [tagged_tweets_2[key][0], {'MISSING'}]
                discrepancies[key][f2] = tagged_tweets_2[key]
                an1_label = set()
                an2_label = tagged_tweets_2[key][1]
            elif not tagged_tweets_2[key]:
                discrepancies[key][f1] = tagged_tweets_1[key]
                discrepancies[key][f2] = [tagged_tweets_1[key][0], {'MISSING'}]
                an1_label = tagged_tweets_1[key][1]
                an2_label = set()
        if an1_label not in labels:
            an1_label = set()
        if an2_label not in labels:
            an2_label = set()
        an1_i = labels.index(an1_label)
        an2_i = labels.index(an2_label)
        ck_mtx[an1_i][an2_i] += 1
    np.savetxt(outfile, ck_mtx, delimiter=",")
    for key in discrepancies:
        textid = list(discrepancies[key][f1][0])[0]
        tweet_text = ' '.join(tweet_dict[textid])
        print("RESOLVE:")
        print("Tweet: {0}".format(tweet_text))
        print("annotator 1 tags {0}".format(discrepancies[key][f1][1]))
        print("annotator 2 tags {0}".format(discrepancies[key][f2][1]))
        unresolved = True
        while unresolved:
            choice = input("Enter annotator number(s): ")
            try:
                val = int(choice.replace(' ', ''))
                if val == 1:
                    results[key] = [tweet_text, discrepancies[key][f1][1]]
                    unresolved = False
                elif val == 2:
                    results[key] = [tweet_text, discrepancies[key][f2][1]]
                    unresolved = False
                elif val == 12:
                    results[key] = [tweet_text, discrepancies[key][f1][1] | discrepancies[key][f2][1]]
                    unresolved = False
            except ValueError:
                print("Need an integer: 1, 2, or 12 for both")
    return results


def write(results, outfile):
    fields = ["tweet_id", "tweet_text", "tweet_classifications"]
    rows = []
    for key in results:
        rows.append([key, results[key][0], '_'.join(list(results[key][1]))])        
    with open(outfile, 'w') as csvfile:
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  
        # writing the fields  
        csvwriter.writerow(fields)  
        # writing the data rows  
        csvwriter.writerows(rows)


def main(f1, f2, gold_outfile, kappa_outfile):
    final = resolve(f1, f2, kappa_outfile)
    write(final, gold_outfile)
