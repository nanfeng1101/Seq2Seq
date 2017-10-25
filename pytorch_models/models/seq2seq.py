# -*- coding:utf-8 -*-
__author__ = 'chenjun'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from beam_search import BeamSearch
from utils.util import *


class Encoder(nn.Module):
    """
    seq2seq encoder use bi-directional gru.
    """
    def __init__(self, embedding, hidden_size, num_layers=1):
        """
        init.
        :param embedding: embedding, (vocab_size, hidden_size)
        :param hidden_size: hidden(input) size for rnn layer
        :param num_layers: num of layers
        """
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.reduce_c_w = Variable(torch.randn(hidden_size*2, hidden_size) * 0.05)
        self.reduce_h_w = Variable(torch.randn(hidden_size * 2, hidden_size) * 0.05)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

    def init_hidden(self, batch_size):
        """
        init hidden
        :return:
        """
        h0 = Variable(torch.zeros(self.num_layers*2, batch_size, self.hidden_size))
        return h0

    def pre_reduce(self, context, hidden):
        """
        bi-direction state to uni-direction state for decoder.
        :param context: encoder_outputs, (batch, len, hidden_size*2)
        :param hidden: hidden state, (2, batch, hidden_size)
        :return:
        """
        context = context.contiguous()
        context_ = torch.mm(context.view(-1, context.size(2)), self.reduce_c_w)  # contiguous
        hidden_ = torch.mm(hidden.view(hidden.size(1), -1), self.reduce_h_w)
        context = context_.view(context.size(0), context.size(1), -1)
        hidden = hidden_.view(1, hidden.size(1), -1)
        return context, hidden

    def forward(self, inputs, hidden):
        """
        encoder.
        :param inputs: variable([[index]])
        :param hidden: hidden state
        :return:
        """
        output = self.embedding(inputs)
        for i in xrange(self.num_layers):
            output, hidden = self.rnn(output, hidden)
        output, hidden = self.pre_reduce(output, hidden)
        return output, hidden


class AttentionDecoder(nn.Module):
    """
    seq2seq decoder with attention.
    """
    def __init__(self, embedding, hidden_size, vocab_size, source_length, num_layers=1, drop_rate=0.1):
        """
        init.
        :param embedding: embedding, (vocab_size, hidden_size)
        :param hidden_size: hidden_size
        :param vocab_size: vocab_size for output
        :param num_layers: num of layers.
        :param source_length: max length of source sentence
        :param drop_rate: dropout rate for decoder input
        """
        super(AttentionDecoder, self).__init__()
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.drop_rate = drop_rate
        self.source_length = source_length

        self.attn = nn.Linear(self.hidden_size * 2, self.source_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.drop_rate)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)

    def init_hidden(self, batch_size):
        """
        init hidden state.
        :return:
        """
        h0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        return h0

    def forward(self, inputs, hidden, context):
        """
        decoder with attention.
        :param inputs: variable([[index]])
        :param hidden: hidden state
        :param context: encoder output, (batch_size, source_length, hidden_size)
        :return:
        """
        embedded = self.dropout(self.embedding(inputs))  # drop  batch*1*hidden
        attn_weights = F.softmax(self.attn(torch.cat((embedded.squeeze(1), hidden[0]), 1)))  # b *（hiddensize * 2）-> b * sequence_length
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), context)  # batch*1*sequence_length + batch*sequence_length*hidden_size -> batch*1*hidden_size
        output = torch.cat((embedded.squeeze(1), attn_applied.squeeze(1)), 1)  # b*(hidden_size*2)
        output = self.attn_combine(output).unsqueeze(1)  # batch*1*hidden_size
        for i in xrange(self.num_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.linear(output.squeeze(1)))
        return output, hidden, attn_weights  # b*vocab_size


class Seq2Seq(object):
    """
    seq2seq model.
    """
    def __init__(self, vocab_size1, vocab_size2, hidden_size, beam_size, source_length, target_length, generate_num=5, num_layers=1, drop_rate=0.1, lr=0.01):
        self.hidden_size = hidden_size
        self.source_length = source_length
        self.target_length = target_length
        self.vocab_size2 = vocab_size2
        self.beam_size = beam_size
        self.generate_num = generate_num
        self.embedding1 = nn.Embedding(vocab_size1, hidden_size)
        self.embedding2 = nn.Embedding(vocab_size2, hidden_size)
        self.encoder = Encoder(self.embedding1, hidden_size, num_layers)
        self.decoder = AttentionDecoder(self.embedding2, hidden_size, vocab_size2, source_length, num_layers, drop_rate)
        self.encoder_optimizer = optim.RMSprop(self.encoder.parameters(), lr)
        self.decoder_optimizer = optim.RMSprop(self.decoder.parameters(), lr)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, input_variable, target_variable, batch_size):
        encoder_hidden = self.encoder.init_hidden(batch_size)
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        encoder_outputs, encoder_hidden = self.encoder(input_variable, encoder_hidden)
        decoder_input = Variable(torch.LongTensor([[SOS_token]]*batch_size))  # SOS
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        decoder_hidden = encoder_hidden
        loss = 0.
        # Teacher forcing: Feed the target as the next input
        for di in xrange(self.target_length):
            target = target_variable[:, di]  # (batch,)
            decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)  # SOS + Target
            decoder_input = target.unsqueeze(1)  # Target, (batch, 1)
            loss += self.criterion(decoder_output, target)
        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        return loss.data[0] / self.target_length

    def generate(self, input_variable, batch_size):
        input_variable = input_variable.view(batch_size, -1)
        encoder_hidden = self.encoder.init_hidden(batch_size)
        encoder_outputs, encoder_hidden = self.encoder(input_variable, encoder_hidden)

        decoder_hidden = encoder_hidden
        decoder_inputs = [(Variable(torch.LongTensor([[SOS_token]])), decoder_hidden)]  # SOS
        beam = BeamSearch(self.vocab_size2, self.beam_size, decoder_hidden)
        # loop beam search
        for di in xrange(self.target_length):
            decoder_outputs = []
            for decoder_input, decoder_hidden in decoder_inputs:
                decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)  # SOS + Predict
                decoder_outputs.append((decoder_output, decoder_hidden))
            decoder_inputs = beam.beam_search(decoder_outputs)
        return beam.generate(self.generate_num)
