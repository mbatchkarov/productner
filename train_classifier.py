"""Script to train a product category classifier based on product titles and descriptions
"""

import sys, csv
from tokenizer import WordTokenizer
from classifier import ProductClassifier
from collections import Counter

MAX_TEXTS = 1000000

def usage():
    print """
USAGE: python train_classifier.py data_file.csv
FORMAT: "title","brand","description","categories"
"""
    sys.exit(0)

def main(argv):
    if len(argv) < 2:
        usage()

    # Fetch data
    texts, categories = [], []
    with open(sys.argv[1], 'rb') as f:
        reader = csv.DictReader(f, fieldnames=["title","brand","description","categories"])
        count = 0
        for row in reader:
            count += 1
            # TODO change here what we train on, and what categories are used
            text, category = row['title'], row['categories'].split(' / ')[0]
            texts.append(text)
            categories.append(category)
            if count >= MAX_TEXTS:
                break
    print('Processed %s texts.' % len(texts))

    tmpx, tmpy = [], []
    c = Counter(categories)
    for x, y in zip(texts, categories):
        if c[y] > 200:
            tmpx.append(x)
            tmpy.append(y)

    texts = tmpx
    categories = tmpy

    print Counter(tmpy)

    # Tokenize texts
    tokenizer = WordTokenizer()
    tokenizer.load()
    data = tokenizer.tokenize(texts)

    # Get labels from classifier
    classifier = ProductClassifier()
    labels = classifier.get_labels(categories)

    # Compile classifier network and train
    classifier.compile(tokenizer)
    classifier.train(data, labels)

if __name__ == "__main__":
    main(sys.argv)
