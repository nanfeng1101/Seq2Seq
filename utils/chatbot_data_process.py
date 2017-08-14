# -*- coding:utf-8 -*-
__author__ = 'chenjun'

import cPickle as pickle
from util import *


def load_train_data_for_theano(data_path):
    encoder_input, decoder_input, decoder_target = [], [], []
    with open(data_path) as mf:
        lines = pickle.load(mf)
        for line in lines:
            [ei, dt, di] = line
            encoder_input.append(pad(ei))
            decoder_input.append(pad(di))
            decoder_target.append(pad(dt))
        encoder_input = np.asarray(encoder_input, dtype='int64')
        decoder_input = np.asarray(decoder_input, dtype='int64')
        decoder_target = np.asarray(decoder_target, dtype='int64')
    return encoder_input, decoder_input, decoder_target


def load_train_data_for_torch(data_path):
    res = []
    data = pickle.load(open(data_path))
    for line in data:
        ei = line[0]  # input
        dt = line[1]  # target
        res.append((np.asarray(ei, dtype="int64"), np.asarray(dt, dtype="int64")))
    return res


def load_train_data_for_tensorflow(data_path):
    encoder_input, encoder_length, decoder_input, decoder_target = [], [], [], []
    with open(data_path) as mf:
        lines = pickle.load(mf)
        for line in lines:
            [ei, dt, di] = line
            encoder_input.append(pad(ei))
            encoder_length.append(len(ei))
            decoder_input.append(pad(di))
            decoder_target.append(pad(dt))
        encoder_length = np.asarray(encoder_length, dtype="int64")
        encoder_input = np.asarray(encoder_input, dtype='int64')
        decoder_input = np.asarray(decoder_input, dtype='int64')
        decoder_target = np.asarray(decoder_target, dtype='int64')
    return encoder_input, encoder_length, decoder_input, decoder_target


def analyze_movie_lines(movie_lines_file, text_mark_file, text_corpus_file):
    text_mark = {}
    text_corpus = []
    with open(movie_lines_file) as f:
        line_data = f.readlines()
        for line in line_data:
            fields = line.strip().split(" +++$+++ ")
            mark, text = fields[0].strip(), clean_text(unicode(fields[-1].strip(), errors="ignore"))
            text_mark[mark] = text
            text_corpus += text,
    with open(text_mark_file, "wb") as f:
        json.dump(text_mark, f)
    with open(text_corpus_file, "wb") as f:
        for line in text_corpus:
            f.write(line + "\n")


def analyze_movie_conversation(movie_conversation_file, text_mark_file, coversation_save_file):
    conversations = []
    mark_text = load_json(text_mark_file)
    with open(movie_conversation_file) as f:
        lines = f.readlines()
        for line in lines:
            marks = line.strip().split(" +++$+++ ")[-1][1:-1]
            marks = marks.strip().split(", ")
            for i in range(len(marks) - 1):
                mark_a, mark_b = marks[i][1:-1], marks[i + 1][1:-1]
                text_a, text_b = mark_text[mark_a], mark_text[mark_b]
                conversation = text_a + " +++$+++ " + text_b
                conversations.append(conversation)
    with open(coversation_save_file, "wb") as f:
        for conversation in conversations:
            f.write(conversation + "\n")


def corpus_word2index(corpus, word2index_file, index2word_file):
    word2index = {"<SOS>": SOS_token, "<EOS>": EOS_token, "<UNK>": UNK_token}  # 0 used for pad
    index2word = {SOS_token: "<SOS>", EOS_token: "<EOS>", UNK_token: "<UNK>"}
    index = 4
    dix = {}
    for sentence in corpus:
        for word in sentence:
            if word not in dix:
                dix[word] = 1
            else:
                dix[word] += 1
    dix = {k: v for k, v in dix.items() if v >= 2}  # filter
    for key in dix:
        word2index[key] = index
        index2word[index] = key
        index += 1

    print "corpus vocab size: ", index
    with open(word2index_file, "wb") as f:
        json.dump(word2index, f)
    with open(index2word_file, "wb") as f:
        json.dump(index2word, f)


def generate_train_data(conversations_file, train_data_save_file):
    a_data = []
    b_data = []
    with open(conversations_file) as f:
        conversations = f.readlines()
        for conversation in conversations:
            try:
                [text_a, text_b] = conversation.strip().split(" +++$+++ ")
                text_a, text_b = tokenize(text_a), tokenize(text_b)
                if len(text_a) < MAX_LENGTH and len(text_b) < MAX_LENGTH:
                    a_data.append(text_a)
                    b_data.append(text_b)
            except Exception, e:
                print e
    assert len(a_data) == len(b_data)
    corpus = a_data + b_data
    print "corpus length: ", len(corpus)
    corpus_word2index(corpus, word2index_file="../data/w2i.json", index2word_file='../data/i2w.json')
    train_data = []
    for i in xrange(len(a_data)):
        encoder_input = sentence2indices(a_data[i], "../data/w2i.json")
        target_input = sentence2indices(b_data[i], "../data/w2i.json")
        decoder_input = [SOS_token] + target_input  ### SOS
        train_data.append([encoder_input, target_input, decoder_input])
    with open(train_data_save_file, "w") as f:
        pickle.dump(train_data, f)


if __name__ == '__main__':
    path = "/Users/chenjun/PycharmProjects/seq2seq/"
    analyze_movie_lines(path+"corpus/cornell_movie_dialogs_corpus/movie_lines.txt", path+"data/mark_text.json", path+"data/corpus.txt")
    analyze_movie_conversation(path+"corpus/cornell_movie_dialogs_corpus/movie_conversations.txt", path+"data/mark_text.json", path+"data/conversations.txt")
    generate_train_data(path+"data/conversations.txt", path+"data/train_data.pkl")

