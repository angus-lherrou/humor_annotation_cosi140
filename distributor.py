import os
import shutil
from typing import List
from collections import namedtuple
from preprocessor import compose_xml
from itertools import cycle

ANNO_DIR = 'annotated_data'
OUT_DIR = 'adjudication'

Annotator = namedtuple('Annotator', 'name count')


def get_next_annotator(ctr: List[Annotator]):
    i = 0
    avoid = -1
    while True:
        if ctr[i].count == 0:
            yield i, ctr[i].name
            avoid = i
            i = (i + 1) % len(ctr)
        else:
            min = (avoid + 1) % len(ctr)
            for j in range(len(ctr)):
                if ctr[j].count < ctr[min].count and j != avoid:
                    min = j
            yield min, ctr[min].name
            avoid = min


def distribute_to_annotators(annotators: list):
    annotator_counts = [Annotator(annotator, 0) for annotator in annotators]
    if not os.path.exists('packages'):
        os.mkdir('packages')
    for annotator in annotators:
        if not os.path.exists("packages/" + annotator):
            os.mkdir("packages/" + annotator)
    next_annotator = get_next_annotator(annotator_counts)
    for file in os.listdir("train_dir/train_data"):
        with open(f"train_dir/train_data/{file}", 'r') as tsv:
            text = tsv.read()
            num_tweets = len(text.splitlines())

        # distribute file to first annotator
        index, annotator = next(next_annotator)
        print(annotator)
        annotator_counts[index] = Annotator(annotator_counts[index].name,
                                            annotator_counts[index].count + num_tweets)
        with open(f"packages/{annotator}/{file[:-4]}.xml", 'w') as xml:
            xml.write(compose_xml(text))

        # distribute file to second annotator
        index, annotator = next(next_annotator)
        print(annotator)
        annotator_counts[index] = Annotator(annotator_counts[index].name,
                                            annotator_counts[index].count + num_tweets)
        with open(f"packages/{annotator}/{file[:-4]}.xml", 'w') as xml:
            xml.write(compose_xml(text))

        if sum(anno.count for anno in annotator_counts) > 3000:
            break

    with open(f"train_dir/train_data/420_Celebs.tsv", 'r') as tsv:
        text = tsv.read()
        num_tweets = len(text.splitlines())
    with open(f"packages/katie/420_Celebs.xml", "w") as celebs:
        celebs.write(compose_xml(text))
    with open(f"packages/jiaying/420_Celebs.xml", "w") as celebs:
        celebs.write(compose_xml(text))

    for annotator in annotator_counts:
        print(num_tweets)
        print(annotator.name, annotator.count)


def distribute_to_adjudicators(adjudicators: list):
    adjudicator = cycle(adjudicators)
    adj_dict = {}

    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    for root, dirs, files in os.walk(ANNO_DIR):
        for name in files:
            if len(dirs) == 0:
                if name in adj_dict:
                    this_adjudicator = adj_dict[name]
                else:
                    this_adjudicator = next(adjudicator)
                    adj_dict[name] = this_adjudicator
                if not os.path.exists(os.path.join(OUT_DIR, this_adjudicator)):
                    os.mkdir(os.path.join(OUT_DIR, this_adjudicator))
                if not os.path.exists(os.path.join(OUT_DIR, this_adjudicator, name[:-4])):
                    os.mkdir(os.path.join(OUT_DIR, this_adjudicator, name[:-4]))
                shutil.copy(os.path.join(root, name),
                            os.path.join(OUT_DIR, this_adjudicator, name[:-4],
                                         f'{os.path.split(root)[-1]}_{name}'))


if __name__ == '__main__':
    # distribute_to_annotators(["katie", "jiaying", "linxuan"])
    distribute_to_adjudicators(["angus", "eli", "joe"])
