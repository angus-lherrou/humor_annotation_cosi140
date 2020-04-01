import os
from typing import List
from collections import namedtuple
from preprocessor import compose_xml

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



if __name__ == '__main__':
    distribute_to_annotators(["katie", "jiaying", "linxuan"])
