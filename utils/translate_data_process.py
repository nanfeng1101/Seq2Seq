# -*- coding:utf-8 -*-
__author__ = 'chenjun'
import cPickle as pickle
import json, re
import numpy as np
from util import *


class Corpus:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {1: "SOS", 2: "EOS"}
        self.n_words = 3  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def read_langs(lang1, lang2, reverse=False):  # eng[1][output] - fra[2][input]
    print("Reading lines...")
    # Read the file and split into lines
    lines = open('../corpus/%s-%s.txt' % (lang1, lang2)).read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[clean_text(s) for s in l.split('\t')] for l in lines]  # eng-fra

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Corpus(lang2)
        output_lang = Corpus(lang1)
    else:
        input_lang = Corpus(lang2)
        output_lang = Corpus(lang1)

    return input_lang, output_lang, pairs


def filter_pair(p):  # eng - fra
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and p[0].startswith(eng_prefixes)


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse)  # eng[lang1][output] - fra[lang2][input]
    print("Read %s sentence pairs" % len(pairs))
    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.add_sentence(pair[1])  # pair: eng - fra
        output_lang.add_sentence(pair[0])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    json.dump(input_lang.index2word, open("../data2/fra_i2w.json", "w"))
    json.dump(output_lang.index2word, open("../data2/eng_i2w.json", "w"))
    json.dump(input_lang.word2index, open("../data2/fra_w2i.json", "w"))
    json.dump(output_lang.word2index, open("../data2/eng_w2i.json", "w"))
    data = []
    for pair in pairs:
        fra = sentence2indices(pair[1], input_lang.word2index)
        eng = sentence2indices(pair[0], output_lang.word2index)
        data.append([fra, eng])
    print data[0]
    pickle.dump(data, open("../data2/train_data.pkl", "w"))


def load_data_for_pytorch(path):
    data = pickle.load(open(path))
    res = []
    for fra, eng in data:
        ei = np.asarray(fra, dtype="int64")  # fra
        tg = np.asarray(eng, dtype="int64")  # eng
        res.append((ei, tg))
    return res


def load_data_for_theano(path):
    data = pickle.load(open(path))
    encoder_input, decoder_input, decoder_target = [], [], []
    for fra, eng in data:
        ei = fra  # fra
        tg = eng  # eng
        di = [SOS_token] + eng
        encoder_input.append(pad(ei))
        decoder_input.append(pad(di))
        decoder_target.append(pad(tg))
    encoder_input = np.asarray(encoder_input, dtype="int64")
    decoder_input = np.asarray(decoder_input, dtype="int64")
    decoder_target = np.asarray(decoder_target, dtype="int64")
    return encoder_input, decoder_input, decoder_target


def load_data_for_tensorflow(data_path):
    encoder_input, encoder_length, decoder_input, decoder_target = [], [], [], []
    with open(data_path) as mf:
        data = pickle.load(mf)
        for fra, eng in data:
            ei = fra  # fra
            tg = eng  # eng
            di = [SOS_token] + eng
            encoder_length.append(len(ei))
            encoder_input.append(pad(ei))
            decoder_input.append(pad(di))
            decoder_target.append(pad(tg))
        encoder_length = np.asarray(encoder_length, dtype="int64")
        encoder_input = np.asarray(encoder_input, dtype='int64')
        decoder_input = np.asarray(decoder_input, dtype='int64')
        decoder_target = np.asarray(decoder_target, dtype='int64')
    return encoder_input, encoder_length, decoder_input, decoder_target


if __name__ == "__main__":
    prepare_data('eng', 'fra')