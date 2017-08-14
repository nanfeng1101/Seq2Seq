# -*- coding:utf-8 -*-
__author__ = 'chenjun'

import theano
import theano.tensor as T
from rnn import GRU, ReLU
from optimizer import Optimizer
from utils.util import *


class Encoder(object):
    """
    seq2seq encoder.
    """
    def __init__(self, rng, embedding, hidden_size, num_layers=1):
        """
        model init.
        :param rng: np random with seed.
        :param embedding: encoder embedding
        :param hidden_size: hidden_size for gru.
        :param num_layers: num of layers.
        """
        self.embedding = embedding
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru_layer = GRU(rng, hidden_size, hidden_size)
        self.params = []
        self.params += self.gru_layer.params

    def __call__(self, inputs, mask, h=None):
        """
        encoder using gru layer
        :param inputs: inputs word indices. (batch_size, max_length).
        :param mask: mask for input.
        :param h: final state
        :return:
        """
        out = self.embedding[inputs.flatten()].reshape((inputs.shape[0], inputs.shape[1], self.hidden_size))
        for i in xrange(self.num_layers):
            out, h = self.gru_layer(out, mask, h)
        return out, h


class Decoder(object):
    """
    seq2seq decoder.
    """
    def __init__(self, rng, embedding, vocab_size, hidden_size, max_length, num_layers=1):
        """
        model init
        :param rng: np random with seed.
        :param embedding: decoder embedding
        :param vocab_size: target vocab_size
        :param hidden_size: hidden size for gru layer
        :param max_length: sequence max length
        :param num_layers: num of layers
        """
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.max_length = max_length
        self.gru_layer = GRU(rng, hidden_size, hidden_size)
        self.linear = theano.shared(value=(rng.randn(hidden_size, vocab_size) * 0.1).astype(theano.config.floatX),
                                    name="linear", borrow=True)

        self.params = [self.linear]
        self.params += self.gru_layer.params

    def __call__(self, inputs, mask, h):
        """
        decoder using gru layer
        :param inputs: input
        :param mask: mask
        :param h: final state
        :return:
        """
        output = self.embedding[inputs.flatten()].reshape((-1, 1, self.hidden_size))  # batch*1*hidden_size
        for i in xrange(self.num_layers):
            output = ReLU(output)
            output, h = self.gru_layer(output, mask, h)
        output = T.tensordot(output, self.linear, axes=[2, 0])  # b*1*hidden_size hidden_size*vocab_size
        return output, h  # b*1*vocab_size(unscaled), b*hidden_size


class AttentionDecoder(object):
    """
    seq2seq decoder with soft-attention.
    """
    def __init__(self, rng, embedding, vocab_size, hidden_size, max_length, num_layers=1):
        """
        model init
        :param rng: np random with seed.
        :param embedding: decoder embedding
        :param vocab_size: target vocab_size
        :param hidden_size: hidden size for gru layer
        :param max_length: sequence max length
        :param num_layers: num of layers
        """
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.max_length = max_length
        self.gru_layer = GRU(rng, hidden_size, hidden_size)
        # params
        self.attn_W = theano.shared(value=(rng.randn(hidden_size*2, max_length) * 0.1).astype(theano.config.floatX), name="attn_W", borrow=True)
        self.attn_b = theano.shared(value=np.zeros(shape=(max_length, ), dtype=theano.config.floatX), name="attn_W", borrow=True)
        self.attn_combine_W = theano.shared(value=(rng.randn(hidden_size*2, hidden_size) * 0.1).astype(theano.config.floatX), name="attn_c_w", borrow=True)
        self.attn_combine_b = theano.shared(value=np.zeros(shape=(hidden_size, ), dtype=theano.config.floatX), name="attn_c_b", borrow=True)
        self.linear = theano.shared(value=(rng.randn(hidden_size, vocab_size) * 0.1).astype(theano.config.floatX), name="linear", borrow=True)
        self.params = [self.attn_W, self.attn_b, self.attn_combine_W,
                       self.attn_combine_b, self.linear]
        self.params += self.gru_layer.params

    def linear_func(self, x, w, b):
        """
        linear function, (x * W + b)
        :param x: input
        :param w: w param
        :param b: bias
        :return:
        """
        linear_out = T.dot(x, w) + b
        return linear_out

    def __call__(self, inputs, mask, h, encoder_outputs):
        """
        decoder using gru layer
        :param inputs: input word indices, (batch_size, 1)
        :param mask: mask for inputs, (batch_size, 1)
        :param h: final state, (batch_size, hidden_size)
        :param encoder_outputs: output of encoder, (batch_size, max_length, hidden_size)
        :return:
        """
        embedded = self.embedding[inputs.flatten()].reshape((-1, self.hidden_size))  # batch*hidden_size
        attn_weights = T.nnet.softmax(self.linear_func(T.concatenate([embedded, h], 1), self.attn_W, self.attn_b))  # batch*(hidden_size*2)-> batch * max_length
        attn_weights = attn_weights.reshape((-1, 1, self.max_length))
        attn_applied = T.batched_dot(attn_weights, encoder_outputs)  # batch*1*max_length   *   batch*max_length*hidden_size -> batch*1*hidden_size
        output = T.concatenate([embedded, attn_applied[:, 0, :]], 1)  # b*(hidden_size*2)
        output = self.linear_func(output, self.attn_combine_W, self.attn_combine_b)  # b*hidden_size
        output = output.reshape((-1, 1, self.hidden_size))
        for i in xrange(self.num_layers):
            output = ReLU(output)
            output, h = self.gru_layer(output, mask, h)
        output = T.tensordot(output, self.linear, axes=[2, 0])
        return output, h, attn_weights  # b*1*vocab_size(unscaled), b*hidden_size, b*max_length


class Seq2SeqTranslate(object):
    """
    seq2seq model for translate.
    """
    def __init__(self, rng, vocab_size_src, vocab_size_tar, hidden_size, max_length, num_layers=1, learning_rate=0.1):
        """
        seq2seq model init.
        :param rng: np random with seed
        :param vocab_size_src: source vocab size
        :param vocab_size_tar: target vocab size
        :param hidden_size: hidden size for gru layer
        :param max_length: sequence max length
        :param num_layers: num of layers
        :param learning_rate: learning rate for updates
        """
        # embedding
        self.src_embedding = theano.shared(
            name="src_embedding",
            value=(rng.randn(vocab_size_src, hidden_size) * 0.1).astype(
                theano.config.floatX),
            borrow=True
        )
        self.tar_embedding = theano.shared(
            name="tar_embedding",
            value=(rng.randn(vocab_size_tar, hidden_size) * 0.1).astype(
                theano.config.floatX),
            borrow=True
        )
        # encoder & decoder & params
        self.encoder = Encoder(rng, self.src_embedding, hidden_size, num_layers)
        self.decoder = AttentionDecoder(rng, self.tar_embedding, vocab_size_tar, hidden_size, max_length, num_layers)
        self.opt = Optimizer()
        self.params = [self.src_embedding, self.tar_embedding]
        self.params += self.encoder.params
        self.params += self.decoder.params
        # place_holder
        self.encoder_inputs = T.lmatrix(name='encoder_input')
        self.encoder_mask = T.lmatrix(name='encoder_mask')
        self.decoder_inputs = T.lmatrix(name='decoder_input')  # SOS + target = teacher force
        self.decoder_mask = T.lmatrix(name='decoder_mask')
        self.decoder_target = T.lmatrix(name='decoder_target')
        # seq2seq train
        encoder_outputs, encoder_h = self.encoder(self.encoder_inputs, self.encoder_mask)
        decoder_outputs = []
        decoder_h = encoder_h
        for i in xrange(max_length):
            word_indices = self.decoder_inputs[:, i].reshape((-1, 1))  # teacher force
            mask = self.decoder_mask[:, i].reshape((-1, 1))
            decoder_out, decoder_h, attn_w = self.decoder(word_indices, mask, decoder_h, encoder_outputs)
            decoder_outputs.append(decoder_out)
        decoder_outputs = T.concatenate(decoder_outputs, 1)  # batch_size * max_length * hidden_size
        softmax_outputs, _ = theano.scan(fn=lambda x: T.nnet.softmax(x), sequences=decoder_outputs)
        batch_cost, _ = theano.scan(fn=self.NLLLoss, sequences=[softmax_outputs, self.decoder_target, self.decoder_mask])
        cost = batch_cost.sum() / self.decoder_mask.sum()
        updates = self.opt.RMSprop(self.params, cost, learning_rate)
        self.train_model = theano.function(
            inputs=[self.encoder_inputs, self.encoder_mask, self.decoder_inputs, self.decoder_mask, self.decoder_target],
            outputs=cost,
            updates=updates
        )
        # seq2seq generate
        generate_outputs = []
        decoder_h = encoder_h
        word_indices = self.decoder_inputs[:, 0].reshape((-1, 1))  # SOS=1
        mask = word_indices  # 1
        for i in xrange(max_length):
            decoder_out, decoder_h, attn_w = self.decoder(word_indices, mask, decoder_h, encoder_outputs)  # predict
            softmax_out = T.nnet.softmax(decoder_out[:, 0, :])
            word_indices = T.cast(T.argmax(softmax_out, -1), dtype="int64")
            generate_outputs.append(word_indices)
        generate_outputs = T.concatenate(generate_outputs, 0)
        self.generate_model = theano.function(
            inputs=[self.encoder_inputs, self.encoder_mask, self.decoder_inputs],
            outputs=generate_outputs
        )

    def NLLLoss(self, pred, y, m):
        """
        negative log likelihood loss.
        :param pred: predict label
        :param y: true label
        :param m: mask
        :return:
        """
        return - (m * T.log(pred)[T.arange(y.shape[0]), y])

    def train(self, encoder_inputs, encoder_mask, decoder_inputs, decoder_mask, decoder_target):
        """
        model train.
        :param encoder_inputs:
        :param encoder_mask:
        :param decoder_inputs:
        :param decoder_mask:
        :param decoder_target:
        :return:
        """
        return self.train_model(encoder_inputs, encoder_mask, decoder_inputs, decoder_mask, decoder_target)

    def generate(self, encoder_inputs, encoder_mask):
        """
        model generate
        :param encoder_inputs:
        :param encoder_mask:
        :return:
        """
        decoder_inputs = np.asarray([[SOS_token]], dtype="int64")
        if encoder_inputs.ndim == 1:
            encoder_inputs = encoder_inputs.reshape((1, -1))
            encoder_mask = encoder_mask.reshape((1, -1))
        rez = self.generate_model(encoder_inputs, encoder_mask, decoder_inputs)
        return rez


class Seq2SeqChatBot(object):
    """
    seq2seq model for chatbot.
    """
    def __init__(self, rng, vocab_size, hidden_size, max_length, num_layers=1, learning_rate=0.1):
        """
        seq2seq model init.
        :param rng: np random with seed
        :param vocab_size:  vocab size
        :param hidden_size: hidden size for gru layer
        :param max_length: sequence max length
        :param num_layers: num of layers
        :param learning_rate: learning rate for updates
        """
        # embedding
        self.embedding = theano.shared(
            name="embedding",
            value=(rng.randn(vocab_size, hidden_size) * 0.1).astype(
                theano.config.floatX),
            borrow=True
        )
        # encoder & decoder & params
        self.encoder = Encoder(rng, self.embedding, hidden_size, num_layers)
        self.decoder = AttentionDecoder(rng, self.embedding, vocab_size, hidden_size, max_length, num_layers)
        self.opt = Optimizer()
        self.params = [self.embedding]
        self.params += self.encoder.params
        self.params += self.decoder.params
        # place_holder
        self.encoder_inputs = T.lmatrix(name='encoder_input')
        self.encoder_mask = T.lmatrix(name='encoder_mask')
        self.decoder_inputs = T.lmatrix(name='decoder_input')  # SOS + target = teacher force
        self.decoder_mask = T.lmatrix(name='decoder_mask')
        self.decoder_target = T.lmatrix(name='decoder_target')
        # seq2seq train
        encoder_outputs, encoder_h = self.encoder(self.encoder_inputs, self.encoder_mask)
        decoder_outputs = []
        decoder_h = encoder_h
        for i in xrange(max_length):
            word_indices = self.decoder_inputs[:, i].reshape((-1, 1))  # teacher force
            mask = self.decoder_mask[:, i].reshape((-1, 1))
            decoder_out, decoder_h, attn_w = self.decoder(word_indices, mask, decoder_h, encoder_outputs)
            decoder_outputs.append(decoder_out)
        decoder_outputs = T.concatenate(decoder_outputs, 1)  # batch_size * max_length * hidden_size
        softmax_outputs, _ = theano.scan(fn=lambda x: T.nnet.softmax(x), sequences=decoder_outputs)
        batch_cost, _ = theano.scan(fn=self.NLLLoss, sequences=[softmax_outputs, self.decoder_target, self.decoder_mask])
        cost = batch_cost.sum() / self.decoder_mask.sum()
        updates = self.opt.RMSprop(self.params, cost, learning_rate)
        self.train_model = theano.function(
            inputs=[self.encoder_inputs, self.encoder_mask, self.decoder_inputs, self.decoder_mask,
                    self.decoder_target],
            outputs=cost,
            updates=updates
        )
        # seq2seq generate
        generate_outputs = []
        decoder_h = encoder_h
        word_indices = self.decoder_inputs[:, 0].reshape((-1, 1))  # SOS=1
        mask = word_indices  # 1
        for i in xrange(max_length):
            decoder_out, decoder_h, attn_w = self.decoder(word_indices, mask, decoder_h, encoder_outputs)  # predict
            softmax_out = T.nnet.softmax(decoder_out[:, 0, :])
            word_indices = T.cast(T.argmax(softmax_out, -1), dtype="int64")
            generate_outputs.append(word_indices)
        generate_outputs = T.concatenate(generate_outputs, 0)
        self.generate_model = theano.function(
            inputs=[self.encoder_inputs, self.encoder_mask, self.decoder_inputs],
            outputs=generate_outputs
        )

    def NLLLoss(self, pred, y, m):
        """
        negative log likelihood loss.
        :param pred: predict label
        :param y: true label
        :param m: mask
        :return:
        """
        return - (m * T.log(pred)[T.arange(y.shape[0]), y])

    def train(self, encoder_inputs, encoder_mask, decoder_inputs, decoder_mask, decoder_target):
        """
        model train.
        :param encoder_inputs:
        :param encoder_mask:
        :param decoder_inputs:
        :param decoder_mask:
        :param decoder_target:
        :return:
        """
        return self.train_model(encoder_inputs, encoder_mask, decoder_inputs, decoder_mask, decoder_target)

    def generate(self, encoder_inputs, encoder_mask):
        """
        model generate
        :param encoder_inputs:
        :param encoder_mask:
        :return:
        """
        decoder_inputs = np.asarray([[SOS_token]], dtype="int64")
        if encoder_inputs.ndim == 1:
            encoder_inputs = encoder_inputs.reshape((1, -1))
            encoder_mask = encoder_mask.reshape((1, -1))
        rez = self.generate_model(encoder_inputs, encoder_mask, decoder_inputs)
        return rez