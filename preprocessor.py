import pickle
import os
from xml.sax.saxutils import quoteattr

with open("ht.pkl", "rb") as pkl:
    categories = pickle.load(pkl)

with open("humor.dtd", "r") as dtd:
    name_vn = dtd.readline().split()[2][1:-1]

xml_start = '<?xml version="1.0" encoding="UTF-8" ?>\n' \
          f'<{name_vn}>\n'
xml_end = f"</{name_vn}>"

allowable_characters = set()
for cat in categories:
    for i in range(1, len(cat)):
        allowable_characters.add(cat[i])


def split_tweet(tweet_to_split: str) -> (str, str, str, str):
    wordlist = tweet_to_split.split()
    tweet_id = wordlist[0]
    score = wordlist[-1]
    wordlist = wordlist[1:-1]
    result = []
    hashtag = ""
    j = 0
    while j < len(wordlist):
        if '#' in wordlist[j]:
            words = wordlist[j].split('#')
            if words[0]:
                wordlist[j] = words[0]
            else:
                wordlist.__delitem__(j)
                j -= 1
            words = words[1:]
            for word in words:
                j += 1
                wordlist.insert(j, '#'+word)
        j += 1

    for word in wordlist:
        if "@midnight" in word.lower():
            continue
        elif word[0] == '#':
            tag = "#"
            word_idx = 1
            while word_idx < len(word) and word[word_idx].lower() in allowable_characters:
                tag += word[word_idx].lower()
                word_idx += 1
            if tag in categories:
                hashtag = tag
            result.append(word)
        else:
            result.append(word)

    return tweet_id, ' '.join(result), hashtag, score


def tag_tweet(tweet_to_tag, span, entity_id):
    tweet_id, tweet_content, hashtag, score = split_tweet(tweet_to_tag)
    tag = f'<tweet id="{entity_id}" tweet_uid="{tweet_id}" spans="{span[0]}~{span[1]}" content={quoteattr(tweet_content)} score="{score}" hashtag="{hashtag}" />'
    return tag


def compose_xml(full_text, tweets=None):
    return xml_start + f'<TEXT><![CDATA[{full_text}]]></TEXT>\n' \
           + (('<TAGS>\n' + '\n'.join(tweets) + '\n</TAGS>\n') if tweets else '') + xml_end


if __name__ == '__main__':
    if not os.path.exists("mae_files"):
        os.mkdir("mae_files")
    for file in os.listdir("train_dir/train_data"):
        with open(f"train_dir/train_data/{file}", 'r') as tsv:
            text = tsv.read().replace("\t", " ")
            idx = 0
            tweet_no = 0
            these_tweets = text.splitlines()
            tagged_tweets = []
            for tweet in these_tweets:
                span_end = idx + len(tweet)
                tagged_tweets.append(tag_tweet(tweet, (idx, span_end), f'tweet{tweet_no}'))
                idx += len(tweet) + 1
                tweet_no += 1
        with open(f"mae_files/{file[:-4]}.xml", 'w') as xml:
            xml.write(compose_xml(text, tagged_tweets))
