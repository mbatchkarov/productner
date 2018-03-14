"""Script to extract product category specific attributes based on product titles and descriptions
"""

import sys, os, csv
import numpy as np
from operator import itemgetter
from tokenizer import WordTokenizer
from classifier import ProductClassifier
from ner import ProductNER

def process(row, tokenizer, classifier, ner):
    """Run a row through processing pipeline

    tokenize -> classify
             -> extract attributes

    Args:
        row (dict(str: str)): Dictionary of field name/field value pairs
        tokenizer (WordTokenizer): Word tokenizer
        classifier (ProductClassifier): Product classifier
    Returns:
        dict(str, float): Dictionary of product categories with associated confidence
        list(list(str)): List of pairs of attribute type and attribute value
    """
    # Classify
    data = tokenizer.tokenize([row['name'] + ' ' + row['description']])
    categories = classifier.classify(data)[0]
    row['category'] = max(list(categories.items()), key=itemgetter(1))[0]
    row['category_prob'] = max(categories.values())

    # Extract entities
    data = tokenizer.tokenize([row['name']])
    tags = ner.tag(data)[0]
    brand, brand_started = '', False
    for word, tag in zip(row['name'].split(' '), tags):
        max_tag = max(list(tag.items()), key=itemgetter(1))[0]
        if max_tag == 'B-B' and (not brand_started):
            brand = word
            brand_started = True
        elif max_tag == 'I-B' and brand_started:
            brand += ' '+word
        else:
            brand_started = False
    row['brand'] = brand
    row['title_tokens'] = data

    return row

def usage():
    print ("""
USAGE: python extract.py model_dir data_file.csv
FORMAT: "id","name","description","price"
""")
    sys.exit(0)

def load_models(model_dir):
    # Load tokenizer
    tokenizer = WordTokenizer()
    tokenizer.load(os.path.join(model_dir, 'tokenizer'))

    # Load classifier
    classifier = ProductClassifier()
    classifier.load(os.path.join(model_dir, 'classifier'))

    # Load named entity recognizer
    ner = ProductNER()
    ner.load(os.path.join(model_dir, 'ner'))

    return tokenizer, classifier, ner

def main(argv):
    if len(argv) < 3:
        usage()
    model_dir = sys.argv[1]
    data_file = sys.argv[2]

    tokenizer, classifier, ner = load_models(model_dir)

    with open(data_file, 'rb') as f:
        reader = csv.DictReader(f)
        outfile = open('.'.join(data_file.split('.')[:-1] + ['processed', 'csv']), 'wb')
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames + ['category', 'category_prob', 'brand'])
        writer.writeheader()
        count = 0
        for row in reader:
            count += 1
            processed_row = process(row, tokenizer, classifier, ner)
            writer.writerow(processed_row)

if __name__ == "__main__":
    main(sys.argv)
