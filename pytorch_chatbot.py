# -*- coding:utf-8 -*-
__author__ = 'chenjun'

import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch_models.seq2seq import Seq2SeqChatBot
from utils.chatbot_data_process import *
import random


def build_model(batch_size=1, vocab_size=11394, hidden_size=128, lr=0.01, n_iters=20000, evaluate=40):
    inputs = load_train_data_for_torch("data/train_data.pkl")
    embedding = nn.Embedding(vocab_size, hidden_size)
    model = Seq2SeqChatBot(embedding, vocab_size, batch_size, hidden_size, sequence_length=MAX_LENGTH, drop_rate=0.1, lr=lr)
    for i in xrange(n_iters):
        input = random.choice(inputs)
        ei = Variable(torch.from_numpy(input[0]))
        dt = Variable(torch.from_numpy(input[1]))
        loss = model.train(ei, dt)
        if i % 50 == 0:
            print ("index: %d, loss: %f") % (i, loss)
    for i in xrange(evaluate):
        input = random.choice(inputs)
        ei = Variable(torch.from_numpy(input[0]))
        generate, attn_w = model.generate(ei)
        print "> ", indices2sentence(input[0], 'data/i2w.json')
        print "= ", indices2sentence(input[1], 'data/i2w.json')
        print "< ", indices2sentence(generate, 'data/i2w.json')
        print ""


if __name__ == '__main__':
    build_model()