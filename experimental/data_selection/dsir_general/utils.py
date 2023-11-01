from nltk.tokenize import WordPunctTokenizer
from nltk import ngrams
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk
import numpy as np
import subprocess
from nltk.corpus import stopwords
import string


nltk.download('stopwords')


def hash_buckets(text, num_buckets=1e4):
    return int(abs(hash(text)) % num_buckets)


wpt = WordPunctTokenizer()


def get_ngram_info(line, n=2, num_buckets=10000):
    words = wpt.tokenize(line.lower())
    # words = line.lower().split()
    counts = np.zeros(num_buckets, dtype=int)
    for w in words:
        counts[hash_buckets(w, num_buckets=num_buckets)] += 1
    for i in range(2, n + 1):
        for ng in list(ngrams(words, i)):
            counts[hash_buckets(ng, num_buckets=num_buckets)] += 1
    return counts


def linecount(filename):
    out = subprocess.Popen(['wc', '-l', filename],
                           stdout=subprocess.PIPE).communicate()[0]
    return int(out.strip().partition(b' ')[0])


stop = set(stopwords.words('english') + list(string.punctuation))
numeric = set(list(string.digits))


def transform_text(text):
    return word_tokenize(text.lower())


def repeating_filter(x_tok, n=1):
    if len(x_tok) == 0:
        return 0
    counts = Counter(x_tok)
    if n == 1:
        ratio = (max(counts.values()) / len(x_tok))
    else:
        ratio = sum(sorted(counts.values(), reverse=True)[:n]) / len(x_tok)
    return ratio


def mostly_uninformative_filter(x_tok):
    if len(x_tok) == 0:
        return 0
    informative_ratio = (len([x for x in x_tok if x not in stop]) / len(x_tok))
    return informative_ratio


def numeric_filter(x_tok):
    if len(x_tok) == 0:
        return 0
    ratio = (len([x for x in x_tok if x not in numeric]) / len(x_tok))
    return ratio
