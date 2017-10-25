# -*- coding:utf-8 -*-
__author__ = 'chenjun'

import torch
from torch.autograd import Variable
from utils.util import *


"""Beam search module.
Beam search takes the top K results from the model, predicts the K results for
each of the previous K result, getting K*K results. Pick the top K results from
K*K results, and start over again until certain number of results are fully
decoded.
"""


class Hypothesis(object):
    """Defines a hypothesis during beam search."""
    def __init__(self, tokens, log_prob, state):
        """Hypothesis constructor.
        Args:
          tokens: start tokens for decoding.
          log_prob: log prob of the start tokens, usually 1.
          state: decoder state.
        """
        self.tokens = tokens
        self.log_prob = log_prob
        self.state = state

    def extend(self, token, log_prob, new_state):
        """Extend the hypothesis with result from latest step.
        Args:
          token: latest token from decoding.
          log_prob: log prob of the latest decoded tokens.
          new_state: decoder output state. Fed to the decoder for next step.
        Returns:
          New Hypothesis with the results from latest step.
        """
        return Hypothesis(self.tokens + [token], self.log_prob + log_prob, new_state)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def sequence_tokens(self):
        return self.tokens

    @property
    def decode_state(self):
        return self.state


class BeamSearch(object):
    """Beam search for generation."""

    def __init__(self, vocab_size, beam_size, state=None):
        """
        beam search init.
        :param vocab_size: target vocab size
        :param beam_size: beam size
        """
        self.beam_size = beam_size
        self.vocab_size = vocab_size
        self.hypothesis = [Hypothesis([], 0.0, state)] * self.beam_size
        self.results = []

    def top_hypothesis(self, hypothesis, normalize=False):
        """
        sort the hypothesis list based on log_probs and length.
        :param hypothesis: list of hypothesis
        :param normalize: bool, normalized by length, only for last search to output
        :return:
        """
        # This length normalization is only effective for the final results.
        if normalize:
            return sorted(hypothesis, key=lambda h: h.log_prob/len(h.tokens), reverse=True)
        else:
            return sorted(hypothesis, key=lambda h: h.log_prob, reverse=True)

    def variable(self, token):
        """
        convert token to torch variable.
        :param token: int
        :return:
        """
        return Variable(torch.LongTensor([[token]]))

    def beam_search(self, inputs):
        """
        beam search to generate sequence.
        :param inputs: list of decoder outputs, (decoder_out, decode_state)
        :return:
        """
        all_hypothesis = []
        for i, (input, state) in enumerate(inputs):
            top_log_probs, top_tokens = input.data.topk(self.vocab_size)
            for j in xrange(self.beam_size*2):
                token = top_tokens[0][j]  # value
                log_prob = top_log_probs[0][j]  # value
                all_hypothesis.append(self.hypothesis[i].extend(token, log_prob, state))
        # Filter and collect any hypotheses that have the end token.
        self.hypothesis = []
        for h in self.top_hypothesis(all_hypothesis):
            if h.latest_token == EOS_token:
                # Pull the hypothesis off the beam if the end token is reached.
                self.results.append(h)
            else:
                # Otherwise continue to the extend the hypothesis.
                self.hypothesis.append(h)
            if len(self.hypothesis) == self.beam_size or len(self.results) == self.beam_size:
                break
        outputs = [(self.variable(hyp.latest_token), hyp.decode_state) for hyp in self.hypothesis]
        return outputs

    def generate(self, num):
        """
        return top num of generated sequence tokens.
        :return:
        """
        generates = [hyp.sequence_tokens for hyp in self.top_hypothesis(self.results, normalize=True)[:num]]
        return generates
