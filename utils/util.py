# -*- coding:utf-8 -*-
__author__ = 'chenjun'
import numpy as np
import re, json


SOS_token = np.int64(1)
EOS_token = np.int64(2)
UNK_token = np.int64(3)
MAX_LENGTH = np.int64(10)
use_cuda = 0
teacher_forcing_ratio = 0.5
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def parse_output(token_indices):
    res = []
    for token in token_indices:
        if token != EOS_token:  # end
            res.append(str(token))
        else:
            break
    res.append(str(EOS_token))
    return res


def get_mask(data):
    mask = (np.not_equal(data, 0)).astype("int64")
    return mask


def indices2sentence(idxs, index2word):
    if not isinstance(index2word, dict):
        index2word = load_json(index2word)
    return " ".join([index2word[str(idx)] for idx in idxs])


def sentence2indices(text, word2index):
    if not isinstance(word2index, dict):
        word2index = load_json(word2index)
    if not isinstance(text, list):
        text = text.split(" ")
    idxs = [word2index.get(token, UNK_token) for token in text]  # default to <unk>
    idxs.append(EOS_token)
    return idxs


def pad(index_data, max_length=MAX_LENGTH):
    pad_data = index_data[: max_length] + [0] * (max_length - len(index_data))
    return pad_data


def reverse(index_data):
    return index_data[::-1]


def tokenize(text):
    text = clean_text(text)
    return text.strip().split(" ")


def load_json(file_path):
    with open(file_path) as json_file:
        res = json.load(json_file)
    return res


def clean_text(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

