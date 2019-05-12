import numpy as np
from collections import Counter, defaultdict
from blingfire import text_to_words
from tqdm import tqdm


class CharIdf:
    def __init__(self, ngrams=4, tokenizer=text_to_words):
        self.ngrams = ngrams
        self.grams = []
        self.gram_to_index = dict()
        self.tokenizer = tokenizer
        self._fit = False

    @property
    def gram_length(self):
        return len(self.grams)

    def _make_grams(self, word):
        "Make char n-grams from words"
        # skip those chars which you know nothing about
        for i in range(len(word)):
            for j in range(i + self.ngrams - 1, i + self.ngrams + 1):
                yield word[i:j]
            yield word

        # yield word

    def __getitem__(self, word, cache={}):
        "Get a word's vector"
        if word not in cache:
            vec = np.zeros(self.gram_length)
            for gram in self._make_grams(word):
                if gram in self.gram_to_index:
                    vec[self.gram_to_index[gram]] += 1
            cache[word] = vec
        return cache[word]

    def fit(self, docs):
        "Learn idfs"
        self._fit = True
        self.idf = defaultdict(int)
        for doc in docs:
            for word in self.tokenizer(doc).split():
                self.idf[word] += 1
                for gram in self._make_grams(word):
                    if gram not in self.gram_to_index:
                        self.gram_to_index[gram] = len(self.grams)
                        self.grams.append(gram)
        print(self.grams)

    def transform(self, docs):
        "Get vectors for list of strings"
        if not self._fit:
            raise Exception("fit first")
        docvecs = np.zeros((len(docs), self.gram_length))
        for index, doc in enumerate(tqdm(docs)):
            for word, count in Counter(self.tokenizer(doc).split()).items():
                v = (self[word] * count) / (1 + self.idf[word])
                docvecs[index] += v
        return docvecs

    def fit_transform(self, docs):
        self.fit(docs)
        return self.transform(docs)
