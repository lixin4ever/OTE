__author__ = 'lixin77'

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializations, regularizers, constraints
import numpy as np

# self defined embedding layer
class MyEmbedding(Layer):

    def __init__(self, input_dim, output_dim,
                 init='uniform', input_length=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None,
                 mask_zero=False,
                 weights=None, dropout=0., winsize=1, **kwargs):
        """

        :param input_dim:
        :param output_dim:
        :param init:
        :param input_length: length of padded sequence,
        :param W_regularizer:
        :param activity_regularizer:
        :param W_constraint:
        :param mask_zero:
        :param weights:
        :param dropout:
        :param winsize: window size for the window-based word representation
        :param kwargs:
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.input_length = input_length
        self.mask_zero = mask_zero
        self.dropout = dropout


        self.W_constraint = constraints.get(W_constraint)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        if 0. < self.dropout < 1.:
            self.uses_learning_phase = True
        self.initial_weights = weights
        kwargs['input_shape'] = (self.input_length,)
        kwargs['input_dtype'] = 'int32'
        super(MyEmbedding, self).__init__(**kwargs)

        # added by lixin
        self.winsize = winsize

    def build(self, input_shape):
        self.W = self.add_weight((self.input_dim, self.output_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
        self.built = True

    def compute_mask(self, x, mask=None):
        """

        :param x: input sequence
        :param mask: Whether or not the input value 0 is a special "padding" value that should be masked out
        :return:
        """
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(x, 0)

    def call(self, x, mask=None):
        if K.dtype(x) != 'int32':
            x = K.cast(x, 'int32')
        if 0. < self.dropout < 1.:
            retain_p = 1. - self.dropout
            B = K.random_binomial((self.input_dim,), p=retain_p) * (1. / retain_p)
            B = K.expand_dims(B)
            W = K.in_train_phase(self.W * B, self.W)
        else:
            W = self.W
        if self.winsize == 1:
            out = K.gather(W, x)
        else:
            # window based representation
            out = K.gather(W, x)
            out = out.reshape((out.shape[0] / self.winsize, self.winsize * out.shape[1]))
        return out

