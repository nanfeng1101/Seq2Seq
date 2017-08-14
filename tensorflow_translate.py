# -*- coding:utf-8 -*-
__author__ = 'chenjun'

import tensorflow as tf
from tensorflow_models.seq2seq import Seq2SeqTranslate
from utils.util import *
from utils.translate_data_process import load_data_for_tensorflow


def build_model(vocab_size_src=4096, vocab_size_tar=2883, hidden_size=128, layers_num=1, batch_size=100, n_epochs=20, evaluate=40):
    encoder_inputs, encoder_length, decoder_inputs, decoder_target = load_data_for_tensorflow("data2/train_data.pkl")
    num_batchs = encoder_inputs.shape[0] // batch_size
    print ("train_batch_num: %d") % (num_batchs)
    emb_src = np.random.uniform(low=-0.1, high=0.1, size=(vocab_size_src, hidden_size))
    emb_tar = np.random.uniform(low=-0.1, high=0.1, size=(vocab_size_tar, hidden_size))
    model = Seq2SeqTranslate(emb_src, emb_tar, vocab_size_src, vocab_size_tar, hidden_size, MAX_LENGTH, layers_num, learning_rate=0.01)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
    epoch = 0
    while epoch < n_epochs:
        epoch += 1
        for batch_index in xrange(num_batchs):
            ei = encoder_inputs[batch_index * batch_size: (batch_index + 1) * batch_size]
            el = encoder_length[batch_index * batch_size: (batch_index + 1) * batch_size]
            di = decoder_inputs[batch_index * batch_size: (batch_index + 1) * batch_size]
            dt = decoder_target[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model.train(sess, ei, el, di, dt)
            if batch_index % 10 == 0:
                print ("epoch: %d/%d, batch: %d/%d, loss: %f") % (epoch, n_epochs, batch_index, num_batchs, loss)
    # evaluate
    for i in xrange(evaluate):
        index = np.random.randint(low=0, high=encoder_inputs.shape[0])
        ei = encoder_inputs[index]
        el = encoder_length[index]
        dt = decoder_target[index]
        generate = model.generate(sess, ei, el)
        print "> ", indices2sentence(parse_output(ei), 'data2/fra_i2w.json')
        print "= ", indices2sentence(parse_output(dt), 'data2/eng_i2w.json')
        print "< ", indices2sentence(parse_output(generate), 'data2/eng_i2w.json')
        print ""


if __name__ == '__main__':
    build_model()