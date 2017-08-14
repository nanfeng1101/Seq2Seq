# -*- coding:utf-8 -*-
__author__ = 'chenjun'

import tensorflow as tf
from tensorflow.contrib import rnn
from utils.util import *


class Encoder(object):
    """
    seq2seq encoder.
    """
    def __init__(self, embedding, hidden_size, num_layers=1):
        """
        init.
        :param embedding: embedding
        :param hidden_size: hidden size
        :param num_layers: num of layers
        """
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru_cell = rnn.GRUCell(self.hidden_size)

    def __call__(self, inputs, sequence_length, state=None):
        """
        gru layer
        :param inputs: word indices, (batch, max_length)
        :param sequence_length: input sequence length
        :param state: final state
        :return:
        """
        out = tf.nn.embedding_lookup(self.embedding, inputs)
        for i in xrange(self.num_layers):
            out, state = tf.nn.dynamic_rnn(self.gru_cell, out, sequence_length=sequence_length, initial_state=state, dtype=tf.float32)
        return out, state


class AttentionDecoder(object):
    """
    seq2seq decoder with attention.
    """
    def __init__(self, embedding, hidden_size, vocab_size, num_layers=1, max_length=MAX_LENGTH, keep_prop=0.9):
        """
        init.
        :param embedding: embedding, (vocab_size, hidden_size)
        :param hidden_size: hidden_size
        :param vocab_size: vocab_size for output
        :param num_layers: num of layers.
        :param max_length: max length of sentence
        :param keep_prop: keep rate for decoder input
        """
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.keep_prop = keep_prop
        self.max_length = max_length
        # params
        self.attn_W = tf.Variable(tf.random_normal(shape=(self.hidden_size*2, self.max_length)) * 0.1)
        self.attn_b = tf.Variable(tf.zeros(shape=(self.max_length, )))
        self.attn_combine_W = tf.Variable(tf.random_normal(shape=(self.hidden_size*2, self.hidden_size)) * 0.1)
        self.attn_combine_b = tf.Variable(tf.zeros(shape=(self.hidden_size,)))
        self.linear = tf.Variable(tf.random_normal(shape=(self.hidden_size, self.vocab_size)) * 0.1)
        # gru
        self.gru_cell = rnn.GRUCell(self.hidden_size)

    def linear_func(self, x, w, b):
        """
        linear function, (x * W + b)
        :param x: x input
        :param w: W param
        :param b: b bias
        :return:
        """
        linear_out = tf.add(tf.matmul(x, w), b)
        return linear_out

    def __call__(self, inputs, encoder_outputs, state):
        """
        attention decoder using gru.
        :param inputs: word indices(batch, )
        :param encoder_outputs: encoder outputs(batch, max_length, hidden_size)
        :param state: final state
        :return:
        """
        embedded = tf.nn.dropout(tf.nn.embedding_lookup(self.embedding, inputs), keep_prob=self.keep_prop)  # batch*hidden_size
        attn_weights = tf.nn.softmax(self.linear_func(tf.concat((embedded, state), 1), self.attn_W, self.attn_b))  # batch *（hiddensize * 2）-> batch * max_length
        attn_applied = tf.matmul(tf.expand_dims(attn_weights, 1), encoder_outputs)  # batch*1*max_length   *   batch*max_length*hidden_size -> batch*1*hidden_size
        output = tf.concat([embedded, attn_applied[:, 0, :]], 1)  # b*(hidden_size*2)
        output = tf.expand_dims(self.linear_func(output, self.attn_combine_W, self.attn_combine_b), 1)  # b*1*hidden_size
        for i in xrange(self.num_layers):
            output = tf.nn.relu(output)
            output, state = tf.nn.dynamic_rnn(self.gru_cell, output, initial_state=state, dtype=tf.float32)
        output = tf.tensordot(output, self.linear, axes=[[2], [0]])  # b*1*hidden_size hidden_size*vocab_size
        return output, state, attn_weights  # b*1*vocab_size(unscaled), b*max_length


class Seq2SeqTranslate(object):
    """
    seq2seq translate model.
    """
    def __init__(self, emb_words_src, emb_words_tar, vocab_size_src, vocab_size_tar, hidden_size, max_length, layers_num=1, learning_rate=0.1):
        """
        init.
        :param emb_words_src: source embedding
        :param emb_words_tar: target embedding
        :param vocab_size_src: source vocab size
        :param vocab_size_tar: target vocab size
        :param hidden_size: hidden size for rnn(gru) layer
        :param max_length: sequence max length
        :param layers_num: number of layers
        :param learning_rate: learning rate
        """
        self.max_length = max_length
        self.vocab_size_src = vocab_size_src
        self.vocab_size_tar = vocab_size_tar
        self.hidden_size = hidden_size
        self.layers_num = layers_num
        self.lr = learning_rate

        with tf.name_scope("place_holder"):
            # placeholder
            self.encoder_inputs = tf.placeholder(shape=(None, self.max_length), dtype=tf.int64)
            self.encoder_length = tf.placeholder(shape=(None, ), dtype=tf.int64)
            self.decoder_inputs = tf.placeholder(shape=(None, self.max_length), dtype=tf.int64)
            self.decoder_target = tf.placeholder(shape=(None, self.max_length), dtype=tf.int64)

        with tf.name_scope("embedding"):
            # embedding
            self.embedding_src = tf.get_variable(name="embedding_src", dtype=tf.float32, shape=(vocab_size_src, hidden_size),
                                                 initializer=tf.constant_initializer(emb_words_src), trainable=True)
            self.embedding_tar = tf.get_variable(name="embedding_tar", dtype=tf.float32, shape=(vocab_size_tar, hidden_size),
                                                 initializer=tf.constant_initializer(emb_words_tar), trainable=True)
        with tf.name_scope("encoder-decoder"):
            self.encoder = Encoder(self.embedding_src, self.hidden_size, self.layers_num)
            self.decoder = AttentionDecoder(self.embedding_tar, self.hidden_size, self.vocab_size_tar, self.layers_num, self.max_length)
        with tf.variable_scope("seq2seq-train"):
            # train
            encoder_outputs, encoder_state = self.encoder(self.encoder_inputs, self.encoder_length)
            tf.get_variable_scope().reuse_variables()  # reuse
            decoder_outputs = []
            decoder_state = encoder_state
            for i in xrange(self.max_length):
                word_indices = self.decoder_inputs[:, i]
                decoder_out, decoder_state, attn_w = self.decoder(word_indices, encoder_outputs, decoder_state)
                decoder_outputs.append(decoder_out)
            decoder_outputs = tf.concat(decoder_outputs, 1)  # b * l * vocab_size_tar
        with tf.name_scope("cost"):
            # cost
            self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=decoder_outputs, labels=self.decoder_target))
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.cost)
        with tf.variable_scope("seq2seq-generate"):
            # generate
            self.generate_outputs = []
            decoder_state = encoder_state
            word_indices = self.decoder_inputs[:, 0]  # SOS
            for i in xrange(self.max_length):
                decoder_out, decoder_state, attn_w = self.decoder(word_indices, encoder_outputs, decoder_state)
                softmax_out = tf.nn.softmax(decoder_out[:, 0, :])
                word_indices = tf.cast(tf.arg_max(softmax_out, -1), dtype=tf.int64)  # b * 1
                self.generate_outputs.append(word_indices)
            self.generate_outputs = tf.concat(self.generate_outputs, 0)

    def train(self, sess, encoder_inputs, encoder_length, decoder_inputs, decoder_target):
        """
        feed data to train seq2seq model.
        :param sess: session
        :param encoder_inputs: encoder inputs
        :param encoder_length: encoder inputs sequence length, (batch, )
        :param decoder_inputs: decoder inputs
        :param decoder_target: decoder target
        :return:
        """
        res = [self.optimizer, self.cost]
        _, cost = sess.run(res,
                           feed_dict={self.encoder_inputs: encoder_inputs,
                                      self.encoder_length: encoder_length,
                                      self.decoder_inputs: decoder_inputs,
                                      self.decoder_target: decoder_target
                                      })
        return cost

    def generate(self, sess, encoder_inputs, encoder_length):
        """
        feed data to generate.
        :param sess: session
        :param encoder_inputs: encoder inputs
        :param encoder_length: encoder inputs sequence length, (batch=1, )
        :return:
        """
        decoder_inputs = np.asarray([[SOS_token] * MAX_LENGTH], dtype="int64")
        if encoder_inputs.ndim == 1:
            encoder_inputs = encoder_inputs.reshape((1, -1))
            encoder_length = encoder_length.reshape((1, ))
        res = [self.generate_outputs]
        generate = sess.run(res,
                            feed_dict={self.encoder_inputs: encoder_inputs,
                                       self.decoder_inputs: decoder_inputs,
                                       self.encoder_length: encoder_length
                                       })[0]
        return generate


class Seq2SeqChatBot(object):
    """
    seq2seq chatbot model.
    """
    def __init__(self, emb_words, vocab_size, hidden_size, max_length, layers_num=1, learning_rate=0.1):
        """
        init.
        :param emb_words: embedding
        :param vocab_size: vocab size
        :param hidden_size: hidden size for rnn(gru) layer
        :param max_length: sequence max length
        :param layers_num: number of layers
        :param learning_rate: learning rate
        """
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.layers_num = layers_num
        self.lr = learning_rate

        with tf.name_scope("place_holder"):
            # placeholder
            self.encoder_inputs = tf.placeholder(shape=(None, self.max_length), dtype=tf.int64)
            self.encoder_length = tf.placeholder(shape=(None, ), dtype=tf.int64)
            self.decoder_inputs = tf.placeholder(shape=(None, self.max_length), dtype=tf.int64)
            self.decoder_target = tf.placeholder(shape=(None, self.max_length), dtype=tf.int64)

        with tf.name_scope("embedding"):
            # embedding
            self.embedding = tf.get_variable(name="embedding", dtype=tf.float32, shape=(vocab_size, hidden_size),
                                                 initializer=tf.constant_initializer(emb_words), trainable=True)
        with tf.name_scope("encoder-decoder"):
            self.encoder = Encoder(self.embedding, self.hidden_size, self.layers_num)
            self.decoder = AttentionDecoder(self.embedding, self.hidden_size, self.vocab_size, self.layers_num, self.max_length)
        with tf.variable_scope("seq2seq-train"):
            # train
            encoder_outputs, encoder_state = self.encoder(self.encoder_inputs, self.encoder_length)
            tf.get_variable_scope().reuse_variables()  # reuse
            decoder_outputs = []
            decoder_state = encoder_state
            for i in xrange(self.max_length):
                word_indices = self.decoder_inputs[:, i]
                decoder_out, decoder_state, attn_w = self.decoder(word_indices, encoder_outputs, decoder_state)
                decoder_outputs.append(decoder_out)
            decoder_outputs = tf.concat(decoder_outputs, 1)  # b * l * vocab_size_tar
        with tf.name_scope("cost"):
            # cost
            self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=decoder_outputs, labels=self.decoder_target))
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.cost)
        with tf.variable_scope("seq2seq-generate"):
            # generate
            self.generate_outputs = []
            decoder_state = encoder_state
            word_indices = self.decoder_inputs[:, 0]  # SOS
            for i in xrange(self.max_length):
                decoder_out, decoder_state, attn_w = self.decoder(word_indices, encoder_outputs, decoder_state)
                softmax_out = tf.nn.softmax(decoder_out[:, 0, :])
                word_indices = tf.cast(tf.arg_max(softmax_out, -1), dtype=tf.int64)  # b * 1
                self.generate_outputs.append(word_indices)
            self.generate_outputs = tf.concat(self.generate_outputs, 0)

    def train(self, sess, encoder_inputs, encoder_length, decoder_inputs, decoder_target):
        """
        feed data to train seq2seq model.
        :param sess: session
        :param encoder_inputs: encoder inputs
        :param encoder_length: encoder inputs sequence length, (batch, )
        :param decoder_inputs: decoder inputs
        :param decoder_target: decoder target
        :return:
        """
        res = [self.optimizer, self.cost]
        _, cost = sess.run(res,
                           feed_dict={self.encoder_inputs: encoder_inputs,
                                      self.encoder_length: encoder_length,
                                      self.decoder_inputs: decoder_inputs,
                                      self.decoder_target: decoder_target
                                      })
        return cost

    def generate(self, sess, encoder_inputs, encoder_length):
        """
        feed data to generate.
        :param sess: session
        :param encoder_inputs: encoder inputs
        :param encoder_length: encoder inputs sequence length, (batch=1, )
        :return:
        """
        decoder_inputs = np.asarray([[SOS_token] * MAX_LENGTH], dtype="int64")
        if encoder_inputs.ndim == 1:
            encoder_inputs = encoder_inputs.reshape((1, -1))
            encoder_length = encoder_length.reshape((1, ))
        res = [self.generate_outputs]
        generate = sess.run(res,
                            feed_dict={self.encoder_inputs: encoder_inputs,
                                       self.decoder_inputs: decoder_inputs,
                                       self.encoder_length: encoder_length
                                       })[0]
        return generate