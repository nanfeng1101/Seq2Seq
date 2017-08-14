# -*- coding:utf-8 -*-
__author__ = 'chenjun'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.autograd import Variable
from utils.chatbot_data_process import *


class Encoder(nn.Module):
    """
    seq2seq encoder.
    """
    def __init__(self, embedding, batch_size, hidden_size, num_layers=1):
        """
        init.
        :param embedding: embedding, (vocab_size, hidden_size)
        :param batch_size: batch_size, 1
        :param hidden_size: hidden(input) size for rnn layer
        :param num_layers: num of layers
        """
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)

    def init_hidden(self):
        """
        init hidden
        :return:
        """
        h0 = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        return h0

    def forward(self, inputs, hidden):
        """
        encoder.
        :param inputs: variable([[index]])
        :param hidden: hidden state
        :return:
        """
        output = self.embedding(inputs).view(1, 1, -1)
        for i in xrange(self.num_layers):
            output, hidden = self.rnn(output, hidden)
        return output, hidden


class Decoder(nn.Module):
    """
    seq2seq decoder.
    """
    def __init__(self, embedding, batch_size, vocab_size, hidden_size, num_layers=1):
        """
        decoder init.
        :param embedding: embedding, (vocab_size, hidden_size)
        :param batch_size: batch_size, 1
        :param vocab_size: vocab_size for output
        :param hidden_size: hidden(input) size for rnn layer
        :param num_layers: num of layers
        """
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = embedding
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)  # linear(hidden_size) -> (vocab_size)
        self.softmax = nn.LogSoftmax()  # softmax -> prop

    def init_hidden(self):
        """
        init hidden.
        :return:
        """
        h0 = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        return h0

    def forward(self, inputs, hidden):
        """
        decoder.
        :param inputs: variable([[index]])
        :param hidden: hidden state
        :return:
        """
        output = self.embedding(inputs).view(1, 1, -1)
        for i in xrange(self.num_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.linear(output[0]))
        return output, hidden


class AttentionDecoder(nn.Module):
    """
    seq2seq decoder with attention.
    """
    def __init__(self, embedding, batch_size, hidden_size, vocab_size, num_layers=1, sequence_length=MAX_LENGTH, drop_rate=0.1):
        """
        init.
        :param embedding: embedding, (vocab_size, hidden_size)
        :param batch_size: batch_size, 1
        :param hidden_size: hidden_size
        :param vocab_size: vocab_size for output
        :param num_layers: num of layers.
        :param sequence_length: max length of sentence
        :param drop_rate: dropout rate for decoder input
        """
        super(AttentionDecoder, self).__init__()
        self.embedding = embedding
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.drop_rate = drop_rate
        self.sequence_length = sequence_length

        self.attn = nn.Linear(self.hidden_size * 2, self.sequence_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.drop_rate)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)

    def init_hidden(self):
        """
        init hidden state.
        :return:
        """
        h0 = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        return h0

    def forward(self, inputs, hidden, encoder_outputs):
        """
        decoder with attention.
        :param inputs: variable([[index]])
        :param hidden: hidden state
        :param encoder_outputs: encoder output, (sequence_length, hidden_size)
        :return:
        """
        embedded = self.dropout(self.embedding(inputs).view(1, 1, -1))  # drop
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)))  # 1 *（hiddensize * 2）-> 1 * sequence_length
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))  # 1*1*sequence_length + 1*sequence_length*hidden_size -> 1*1*hidden_size
        output = torch.cat((embedded[0], attn_applied[0]), 1)  # 1*(hidden_size*2)
        output = self.attn_combine(output).unsqueeze(0)  # 1*1*hidden_size
        for i in xrange(self.num_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.linear(output[0]))
        return output, hidden, attn_weights  # 1*vocab_size


class Seq2SeqChatBot(object):
    """
    seq2seq model for chatbot.
    """
    def __init__(self, embedding, vocab_size, batch_size, hidden_size, num_layers=1, sequence_length=MAX_LENGTH, drop_rate=0.1, lr=0.01):
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.encoder = Encoder(embedding, batch_size, hidden_size, num_layers)
        self.decoder = AttentionDecoder(embedding, batch_size, hidden_size, vocab_size, num_layers, sequence_length, drop_rate)
        self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr)
        self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr)
        self.criterion = nn.NLLLoss()

    def train(self, input_variable, target_variable):
        input_variable = input_variable.view(-1, 1)
        target_variable = target_variable.view(-1, 1)
        encoder_hidden = self.encoder.init_hidden()  # init encoder_hidden
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]
        encoder_outputs = Variable(torch.zeros(self.sequence_length, self.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
        loss = 0.
        for ei in xrange(input_length):
            encoder_output, encoder_hidden = self.encoder(input_variable[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0][0]  # 1*1*128
        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        decoder_hidden = encoder_hidden  # decoder hidden, from encoder_hidden
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)  # SOS + Target
                loss += self.criterion(decoder_output[0], target_variable[di])
                decoder_input = target_variable[di]  # Target
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)  # SOS + Predict
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = Variable(torch.LongTensor([[ni]]))  # Predict
                decoder_input = decoder_input.cuda() if use_cuda else decoder_input

                loss += self.criterion(decoder_output[0], target_variable[di])
                if ni == EOS_token:
                    break
        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        return loss.data[0] / target_length

    def generate(self, input_variable):
        input_variable = input_variable.view(-1, 1)
        input_length = input_variable.size()[0]
        encoder_hidden = self.encoder.init_hidden()
        encoder_outputs = Variable(torch.zeros(self.sequence_length, self.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
        for ei in xrange(input_length):
            encoder_output, encoder_hidden = self.encoder(input_variable[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]
        decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        decoder_hidden = encoder_hidden  # from encoder
        decoded_words = []
        decoder_attentions = torch.zeros(self.sequence_length, self.sequence_length)
        # loop
        for di in xrange(self.sequence_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)  # SOS + Predict
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == EOS_token:
                decoded_words.append(EOS_token)
                break
            else:
                decoded_words.append(ni)

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        return decoded_words, decoder_attentions[:di + 1]


class Seq2SeqTranslate(object):
    """
    seq2seq model for translate.
    """
    def __init__(self, embedding1, embedding2, vocab_size2, batch_size, hidden_size, num_layers=1, sequence_length=MAX_LENGTH, drop_rate=0.1, lr=0.01):
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.encoder = Encoder(embedding1, batch_size, hidden_size, num_layers)
        self.decoder = AttentionDecoder(embedding2, batch_size, hidden_size, vocab_size2, num_layers, sequence_length, drop_rate)
        self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr)
        self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr)
        self.criterion = nn.NLLLoss()

    def train(self, input_variable, target_variable):
        input_variable = input_variable.view(-1, 1)
        target_variable = target_variable.view(-1, 1)
        encoder_hidden = self.encoder.init_hidden()
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]
        encoder_outputs = Variable(torch.zeros(self.sequence_length, self.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
        loss = 0.
        for ei in xrange(input_length):
            encoder_output, encoder_hidden = self.encoder(input_variable[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0][0]  # 1*1*hidden_size
        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        decoder_hidden = encoder_hidden
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)  # SOS + Target
                loss += self.criterion(decoder_output[0], target_variable[di])
                decoder_input = target_variable[di]  # Target
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)  # SOS + Predict
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = Variable(torch.LongTensor([[ni]]))  # Predict
                decoder_input = decoder_input.cuda() if use_cuda else decoder_input

                loss += self.criterion(decoder_output[0], target_variable[di])
                if ni == EOS_token:
                    break
        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        return loss.data[0] / target_length

    def generate(self, input_variable):
        input_variable = input_variable.view(-1, 1)
        input_length = input_variable.size()[0]
        encoder_hidden = self.encoder.init_hidden()
        encoder_outputs = Variable(torch.zeros(self.sequence_length, self.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
        for ei in xrange(input_length):
            encoder_output, encoder_hidden = self.encoder(input_variable[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0][0]
        decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        decoder_hidden = encoder_hidden
        decoded_words = []
        decoder_attentions = torch.zeros(self.sequence_length, self.sequence_length)
        # loop
        for di in xrange(self.sequence_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)  # SOS + Predict
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == EOS_token:
                decoded_words.append(EOS_token)
                break
            else:
                decoded_words.append(ni)
            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        return decoded_words, decoder_attentions[:di + 1]




