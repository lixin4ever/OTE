__author__ = 'lixin77'

from data import Token, Sequence
from numpy import empty, zeros, int32, log, exp, sum, add
from collections import defaultdict


def logexpsum(a):
    """
    calculate score: log(exp(a).sum())
    :param a:
    :return:
    """
    return log(exp(a).sum())


class CRF(object):
    """
    Self-implemented conditional random fields
    """
    def __init__(self, O, S):
        """
        build model from Observation features and State features
        :param O: observation feature (word-based features), e.g., xt is capital, prefix of xt is ``irr''
        :param S: state features (label-based features), e.g., yt is T
        :return:
        """
        self.Observations = O
        self.States = S
        self.N_O = len(O)
        self.N_S = len(S)
        # weight (parameter) vector of CRF model
        self.W = zeros(self.N_O)

    def log_potentials(self, x):
        """

        :param x:
        :return:
        """
        assert isinstance(x, Sequence)
        N = x.N
        W = self.W
        N_S = self.n_S
        f = x.feature_table
        g0 = empty(N_S)
        g = empty((N - 1, N_S, N_S))
        for y in xrange(N_S):
            g0[y] = W[f[0, None, y]].sum()
        for t in xrange(1, N):
            for y in xrange(N_S):
                for y_1 in xrange(N_S):
                    g[t-1, y_1, y] = W[f[t, y_1, y]].sum()
        return (g0, g)

    def __call__(self, x):
        """
        infer the most probable labellings of ``x''
        :param x:
        :return:
        """
        assert isinstance(x, Sequence)
        self.preprocess([x])
        return self.argmax(x)

    def argmax(self, x):
        """
        Viterbi decoding
        :param x:
        :return:
        """
        assert isinstance(x, Sequence)
        # length of sequence
        N = x.N
        # number of valid labels
        N_O = self.N_O
        g0, g = self.log_potentials(x)
        # backtrace matrix
        back_trace = empty((N, N_O), dtype=int32) * -1
        V = g0
        for t in xrange(1, N):
            U = empty(N_O)
            for y in xrange(N_O):
                w = V + g[t-1, :, y]
                back_trace[t, y] = b = w.argmax()
                U[y] = w[b]
            V = U
        y = V.argmax()
        path = []
        for t in reversed(xrange(N)):
            path.append(y)
            # perform back tracking, return the previous node / label on the current optimal path
            y = back_trace[t, y]
        # the optimal path
        path.reverse()
        return path

    def likelihood(self, x):
        """

        :param x:
        :return:
        """
        assert isinstance(x, Sequence)
        N = x.N
        N_S = self.N_S
        W = self.W
        g0, g = self.log_potentials(x)
        alpha = self.forward(g0, g, N, N_S)
        # sum of the score of all possible sequences
        logZ = logexpsum(alpha[N - 1])
        # return the log-probability
        return sum(W[k] for k in x.target_features) - logZ

    def preprocess(self, x):
        pass


    def forward(self, g0, g, N, N_S):
        """
        calculate forward probabilities
        :param g0:
        :param g:
        :param N: sequence length
        :param N_S: number of valid states (i.e., labels)
        :return: alpha with shape (N, N_S),
        """
        # forward log-probabilities
        alpha = zeros((N, N_S))
        alpha[0, :] = g0
        for t in xrange(1, N):
            # alpha[t-1]
            alpha_t_1 = alpha[t-1, :]
            for y in xrange(N_S):
                # alpha[t, y] is the sum of score of all sequences from 0 to t time and current state (label) is y
                alpha[t, y] = logexpsum(alpha_t_1 + g[t-1, :, y])
        return alpha

    def backward(self, g, N, N_S):
        """
        calculate backward probabilities
        :param g:
        :param N: sequence length
        :param N_S: number of valid states
        :return:
        """
        beta = zeros((N, N_S))
        for t in reversed(xrange(N - 1)):
            # beta[t+1]
            beta_t1 = beta[t+1, :]
            for y in xrange(N_S):
                beta[t, y] = logexpsum(beta_t1 + g[t+1, y, :])
        return beta

    def expectation(self, x):
        """
        calculate the expectation of the sufficient statistics given ``x'' and current parameter settings
        :param x: input sequence
        :return:
        """
        assert isinstance(x, Sequence)
        N = x.N
        N_S = self.N_S
        f = x.feature_table

        g0, g = self.log_potentials(x)
        # forward probabilities
        alpha = self.forward(g0, g, N, N_S)
        # backward probabilities
        beta = self.backward(g, N, N_S)
        # normalization factor
        logZ = logexpsum(alpha[N-1, :])

        E = defaultdict(float)

        c = exp(g0 + beta[0, :] - logZ).clip(0.0, 1.0)

        for y in xrange(N_S):
            p = c[y]
            for k in f[0, None, y]:
                E[k] += p

        for t in xrange(1, N):
            # vectorized computation of the marginal for this transition factor
            c = exp((add.outer(alpha[t-1, :], beta[t, :]) + g[t-1, :, :] - logZ)).clip(0.0, 1.0)

            for yt_1 in xrange(N_S):
                for yt in xrange(N_S):
                    p = c[yt_1, yt]
                    # expectation of this factor is p * f(yt_1, yt, xt, t), f is feature function
                    for k in f[t, yt_1, yt]:
                        E[k] += p
        return E

    def path_features(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """
        assert isinstance(x, Sequence)
        F = x.feature_table
        f = list(F[0, None, y[0]])
        f.extend(k for t in xrange(1, x.N) for k in F[t, y[t-1], y[t]])

    def preprocess(self, data):
        """
        data preprocessing
        :param data:
        :return:
        """
        SS = self.States
        OO = self.Observations
        # (n_states + n_observations) * n_observations
        size = (len(SS) + len(OO)) * len(SS)
        assert self.W.shape[0] == size

        for x in data:
            assert isinstance(x, Sequence)
            if x.feature_table is None:
                pass


