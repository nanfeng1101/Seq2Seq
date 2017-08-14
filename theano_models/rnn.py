# -*- coding:utf-8 -*-
__author__ = 'chenjun'

import theano.tensor as T
import theano
import numpy as np


def ReLU(x):
    y = T.maximum(0.0, x)
    return y


def weight_init(rng, type, shape):
    if type == "glorot":  # tanh | sigmoid 4*
        scale = np.sqrt(6.0 / (shape[0] + shape[1]))
        return rng.uniform(size=shape, low=-scale, high=scale).astype(theano.config.floatX)
    if type == "ortho":
        W = rng.randn(shape[0], shape[0])
        u, s, v = np.linalg.svd(W)
        return u.astype(theano.config.floatX)
    if type == "zero":
        return np.zeros(shape=shape, dtype=theano.config.floatX).astype(theano.config.floatX)
    if type == "uniform":
        return rng.uniform(size=shape, low=-0.1, high=0.1).astype(theano.config.floatX)


class LSTM(object):
    def __init__(self, rng, n_in, n_hidden, activation=T.tanh, inner_activation=T.nnet.sigmoid):
        self.activation = activation
        self.inner_activation = inner_activation
        self.n_in = n_in
        self.n_hidden = n_hidden
        # input gate
        self.W_i = theano.shared(value=weight_init(rng=rng, type="glorot", shape=(n_in, n_hidden)), name="W_i",
                                 borrow=True)
        self.U_i = theano.shared(value=weight_init(rng=rng, type="ortho", shape=(n_hidden,)), name="U_i", borrow=True)
        self.b_i = theano.shared(value=weight_init(rng=rng, type="zero", shape=(n_hidden,)), name="b_i", borrow=True)
        # forget gate
        self.W_f = theano.shared(value=weight_init(rng=rng, type="glorot", shape=(n_in, n_hidden)), name="W_f",
                                 borrow=True)
        self.U_f = theano.shared(value=weight_init(rng=rng, type="ortho", shape=(n_hidden,)), name="U_f", borrow=True)
        self.b_f = theano.shared(value=weight_init(rng=rng, type="zero", shape=(n_hidden,)), name="b_f", borrow=True)
        # memory
        self.W_c = theano.shared(value=weight_init(rng=rng, type="glorot", shape=(n_in, n_hidden)), name="W_c",
                                 borrow=True)
        self.U_c = theano.shared(value=weight_init(rng=rng, type="ortho", shape=(n_hidden,)), name="U_c", borrow=True)
        self.b_c = theano.shared(value=weight_init(rng=rng, type="zero", shape=(n_hidden,)), name="b_c", borrow=True)
        # output gate
        self.W_o = theano.shared(value=weight_init(rng=rng, type="glorot", shape=(n_in, n_hidden)), name="W_o",
                                 borrow=True)
        self.U_o = theano.shared(value=weight_init(rng=rng, type="ortho", shape=(n_hidden,)), name="U_o", borrow=True)
        self.b_o = theano.shared(value=weight_init(rng=rng, type="zero", shape=(n_hidden,)), name="b_o", borrow=True)

        self.params = [self.W_i, self.U_i, self.b_i,
                       self.W_f, self.U_f, self.b_f,
                       self.W_c, self.U_c, self.b_c,
                       self.W_o, self.U_o, self.b_o,
                       ]

    def __call__(self, input, input_lm=None, h0=None, c0=None):
        batch_size = input.shape[0]  # [batch_size, max_length，hidden_size]
        if h0 == None:
            h0 = T.alloc(np.asarray(0., dtype=theano.config.floatX), batch_size, self.n_hidden)
        if c0 == None:
            c0 = T.alloc(np.asarray(0., dtype=theano.config.floatX), batch_size, self.n_hidden)
        if input_lm == None:
            def step(x_t, h_tm_prev, c_tm_prev):
                x_i = T.dot(x_t, self.W_i) + self.b_i
                x_f = T.dot(x_t, self.W_f) + self.b_f
                x_c = T.dot(x_t, self.W_c) + self.b_c
                x_o = T.dot(x_t, self.W_o) + self.b_o

                i_t = self.inner_activation(x_i + T.dot(h_tm_prev, self.U_i))
                f_t = self.inner_activation(x_f + T.dot(h_tm_prev, self.U_f))
                c_t = f_t * c_tm_prev + i_t * self.activation(x_c + T.dot(h_tm_prev, self.U_c))  # internal memory
                o_t = self.inner_activation(x_o + T.dot(h_tm_prev, self.U_o))
                h_t = o_t * self.activation(c_t)  # actual hidden state

                h_t = T.cast(h_t, dtype=theano.config.floatX)
                c_t = T.cast(c_t, dtype=theano.config.floatX)
                return [h_t, c_t]

            self.output, _ = theano.scan(step,
                                         sequences=input.dimshuffle(1, 0, 2),
                                         outputs_info=[h0, c0])
        else:
            def step(x_t, mask, h_tm_prev, c_tm_prev):
                x_i = T.dot(x_t, self.W_i) + self.b_i
                x_f = T.dot(x_t, self.W_f) + self.b_f
                x_c = T.dot(x_t, self.W_c) + self.b_c
                x_o = T.dot(x_t, self.W_o) + self.b_o

                i_t = self.inner_activation(x_i + T.dot(h_tm_prev, self.U_i))
                f_t = self.inner_activation(x_f + T.dot(h_tm_prev, self.U_f))
                c_t = f_t * c_tm_prev + i_t * self.activation(x_c + T.dot(h_tm_prev, self.U_c))  # internal memory
                o_t = self.inner_activation(x_o + T.dot(h_tm_prev, self.U_o))
                h_t = o_t * self.activation(c_t)  # actual hidden state

                h_t = mask * h_t + (1 - mask) * h_tm_prev
                c_t = mask * c_t + (1 - mask) * c_tm_prev
                h_t = T.cast(h_t, dtype=theano.config.floatX)
                c_t = T.cast(c_t, dtype=theano.config.floatX)
                return [h_t, c_t]

            self.output, _ = theano.scan(step,
                                         sequences=[input.dimshuffle(1, 0, 2), T.addbroadcast(input_lm.dimshuffle(1, 0, 'x'), -1)],
                                         outputs_info=[h0, c0])

        h, c = self.output  # [max_length, batch_size,  hidden_size]
        h, c = h.dimshuffle(1, 0, 2), c.dimshuffle(1, 0, 2)
        return h, h[:, -1, :], c[:, -1, -1]


class GRU(object):
    def __init__(self, rng, n_in, n_hidden, activation=T.tanh, inner_activation=T.nnet.sigmoid):
        self.activation = activation
        self.inner_activation = inner_activation
        self.n_hidden = n_hidden
        # update gate
        self.W_z = theano.shared(value=weight_init(rng=rng, type="glorot", shape=(n_in,n_hidden)), name="W_z",borrow=True)
        self.U_z = theano.shared(value=weight_init(rng=rng, type="ortho", shape=(n_hidden,)), name="U_z", borrow=True)
        self.b_z = theano.shared(value=weight_init(rng=rng, type="zero", shape=(n_hidden,)), name="b_z", borrow=True)
        # reset gate
        self.W_r = theano.shared(value=weight_init(rng=rng, type="glorot",shape=(n_in,n_hidden)), name="W_r", borrow=True)
        self.U_r = theano.shared(value=weight_init(rng=rng, type="ortho", shape=(n_hidden,)), name="U_r", borrow=True)
        self.b_r = theano.shared(value=weight_init(rng=rng, type="zero", shape=(n_hidden,)), name="b_r", borrow=True)
        #
        self.W_h = theano.shared(value=weight_init(rng=rng, type="glorot", shape=(n_in, n_hidden)), name="W_h", borrow=True)
        self.U_h = theano.shared(value=weight_init(rng=rng, type="ortho", shape=(n_hidden,)), name="U_h", borrow=True)
        self.b_h = theano.shared(value=weight_init(rng=rng, type="zero", shape=(n_hidden,)), name="b_h", borrow=True)

        self.params = [self.W_z, self.U_z, self.b_z,
                       self.W_r, self.U_r, self.b_r,
                       self.W_h, self.U_h, self.b_h,
                       ]

    def __call__(self, input, input_lm=None, h0=None):
        batch_size = input.shape[0]
        if h0 == None:
            h0 = T.alloc(np.asarray(0., dtype=theano.config.floatX), batch_size, self.n_hidden)
        if input_lm == None:
            def step(x_t, h_tm_prev):
                x_z = T.dot(x_t, self.W_z) + self.b_z
                x_r = T.dot(x_t, self.W_r) + self.b_r
                x_h = T.dot(x_t, self.W_h) + self.b_h

                z_t = self.inner_activation(x_z + T.dot(h_tm_prev, self.U_z))
                r_t = self.inner_activation(x_r + T.dot(h_tm_prev, self.U_r))
                hh_t = self.activation(x_h + T.dot(r_t * h_tm_prev, self.U_h))
                h_t = (1 - z_t) * hh_t + z_t * h_tm_prev
                h_t = T.cast(h_t, dtype=theano.config.floatX)
                return h_t

            self.output, _ = theano.scan(step, sequences=input.dimshuffle(1, 0, 2), outputs_info=h0)
        else:
            def step(x_t, mask, h_tm_prev):
                x_z = T.dot(x_t, self.W_z) + self.b_z
                x_r = T.dot(x_t, self.W_r) + self.b_r
                x_h = T.dot(x_t, self.W_h) + self.b_h
                z_t = self.inner_activation(x_z + T.dot(h_tm_prev, self.U_z))
                r_t = self.inner_activation(x_r + T.dot(h_tm_prev, self.U_r))

                hh = self.activation(x_h + T.dot(r_t * h_tm_prev, self.U_h))
                h_t = z_t * h_tm_prev + (1 - z_t) * hh
                h_t = mask * h_t + (1 - mask) * h_tm_prev
                h_t = T.cast(h_t, dtype=theano.config.floatX)
                return h_t

            self.output, _ = theano.scan(step, sequences=[input.dimshuffle(1, 0, 2), T.addbroadcast(input_lm.dimshuffle(1, 0, 'x'), -1)], outputs_info=h0)
        h = self.output  # [max_length, batch_size, hidden_size]
        h = h.dimshuffle(1, 0, 2)
        return h, h[:, -1, :]