f__author__ = 'lixin77'

from data import Token, Sequence
from numpy import empty, zeros, int32, log, exp, sum, add
from collections import defaultdict
from operator import itemgetter
from functools import partial
from utils import *
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import random
import string
from keras.layers import Conv1D, LSTM, Dense, Embedding, Dropout, Activation
from keras.models import Sequential
from nltk.corpus import stopwords 


delset = string.punctuation
stopws = stopwords.words('english')
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

class AveragedPerceptron(object):
    """
    averaged perceptron extractor
    """
    def __init__(self, t=0, order=2):
        """
        construct function
        :param t: time step
        :return:
        """
        # timesteps
        self.t = t
        # parameters of the model
        self.weights = defaultdict(partial(defaultdict, Weight))
        self.order = order
        self.classes = []
        # number of classes / tags
        self.n_class = 0
        # number of iterations
        self.n_iter = 20
        self.alpha = 1
        # number of updates elapsed
        self.time = 0

        self.tmp_flag = True

    def fit(self, trainset):
        """
        :param trainset: training set
        :return:
        """
        Y = [sent2tags(sent) for sent in trainset]
        print Y[0]
        self.classes = list(set(np.hstack(Y)))
        self.n_class = len(self.classes)
        print "begin training..."
        for i in xrange(self.n_iter):
            # shuffle the sentence order
            random.shuffle(trainset)
            for sent in trainset:
                Y_gold = sent['tags']
                words = sent['words']
                postags = sent['postags']
                Y_pred, Phi = self.predict(words=words, postags=postags)
                #if self.tmp_flag:
                #    print Phi
                #    self.tmp_flag = False
                for (y, y_pred, phi) in zip(Y_gold, Y_pred, Phi):
                    if y != y_pred:
                        # update the parameter
                        self.update(y_gold=y, y_pred=y_pred, feats=phi)
                self.time += 1
        # averaging the historical parameters. (Totally, n_iter * n_train weight vectors are generated)
        self.average()

    def predict(self, words, postags):
        """
        Predict tags given the observation features, e.g., words, part-of-speech tags and any others
        :param words:
        :param postags:
        :return:
        """
        tags_pred = []
        Phi = []
        for phi in self.f_observe(words, postags):
            phi = phi + self.f_transition(tags=tags_pred[-self.order:])
            y_pred = max(self.scores(phi).items(), key=itemgetter(1))[0]
            tags_pred.append(y_pred)
            Phi.append(phi)
        return tags_pred, Phi

    def infer(self, testset):
        """
        Predict tags given the testing sentences, the same as predict
        :param testset: testing dataset
        :return:
        """
        Y_pred = []
        for sent in testset:
            words = sent2words(sent)
            postags = sent2postags(sent)
            tags_pred, _ = self.predict(words=words, postags=postags)
            Y_pred.append(tags_pred)
        return Y_pred

    def scores(self, feat):
        """
        predict the tag based on observation and transition features
        :param feat: derived observation and transition features
        :return:
        """
        p_y_x = dict.fromkeys(self.classes, 0)
        for f in feat:
            for (y, weight) in self.weights[f].items():
                p_y_x[y] += weight.get()
        #print p_y_x
        return p_y_x

    def update(self, y_gold, y_pred, feats):
        """
        update weight vector of the model
        :param y_gold:
        :param y_pred:
        :param feats: features derived from single token and its contexts
        :return:
        """
        for f in feats:
            ptr = self.weights[f]
            # update weight objects
            assert isinstance(ptr[y_gold], Weight)
            ptr[y_gold].update(self.alpha, self.time)
            ptr[y_pred].update(-self.alpha, self.time)

    def f_observe(self, words, postags):
        """

        :param words:
        :param postags:
        :return:
        """
        return [list(self.f_observe_i(words, postags, i)) for i in xrange(len(words))]

    def f_observe_i(self, words, postags, i):
        """
        get observed features by following the feature template in http://www.aclweb.org/anthology/W02-1001
        :return:
        """
        assert len(words) == len(postags)
        yield 'word.identity=%s' % words[i]
        yield 'word.postag=%s' % postags[i]
        n_words = len(words)
        if i == 0:
            yield 'word.BOS=True'
        if i == n_words - 1:
            yield 'word.EOS=True'
        if i >= 1:
            # word identity at i-1 time
            yield 'word.identity@-1=%s' % words[i-1]
            yield 'word.identity@-1=%s, word.identity=%s' % (words[i-1], words[i])
            # pos tag features
            yield 'word.postag@-1=%s' % postags[i-1]
            yield 'word.postag@-1=%s, word.postag=%s' % (postags[i-1], postags[i])
        if i >= 2:
            # word identity at i-2 time
            yield 'word.identity@-2=%s' % words[i-2]
            yield 'word.identity@-2=%s, word.identity@-1=%s' % (words[i-2], words[i-1])

            yield 'word.postag@-2=%s' % postags[i-2]
            yield 'word.postag@-2=%s, word.postag@-1=%s' % (postags[i-2], postags[i-1])

            # trigram postag features
            yield 'word.postag@-2=%s, word.postag@-1=%s, word.postag=%s' % (postags[i-2], postags[i-1], postags[i])
        if i < n_words - 1:
            # word identity at i+1 time
            yield 'word.identity@+1=%s' % words[i+1]
            yield 'word.identity=%s, word.identity@+1=%s' % (words[i], words[i+1])

            yield 'word.postag@+1=%s' % postags[i+1]
            yield 'word.postag=%s, word.postag@+1=%s' % (postags[i], postags[i+1])

        if i < n_words - 2:
            # word identity at i+2 time
            yield 'word.identity@+2=%s' % words[i+2]
            yield 'word.identity@+1=%s, word.identity@+2=%s' % (words[i+1], words[i+2])

            yield 'word.postag@+2=%s' % postags[i+2]
            yield 'word.postag@+1=%s, word.postag@+2=%s' % (postags[i+1], postags[i+2])
            # trigram postag features
            yield 'word.postag=%s, word.postag@+1=%s, word.postag@+2=%s' % (postags[i], postags[i+1], postags[i+2])

    def f_transition(self, tags):
        """
        yield label / transition features
        :param tags: tag sequence
        :return:
        """
        feats = []
        if not tags:
            return []
        n_tag = len(tags)
        for i in xrange(n_tag):
            uni_f = 'word.tag@-%s=%s' % (n_tag - i, tags[i])
            feats.append(uni_f)
            if i < n_tag - 1:
                bi_f = 'word.tag@-%s=%s, word.tag@-%s=%s' % (n_tag - i, tags[i], n_tag - i - 1, tags[i+1])
                feats.append(bi_f)
        return feats

    def average(self):
        """

        :return:
        """
        for (f, clsweights) in self.weights.items():
            for (y, weight) in clsweights.items():
                weight.average(self.time)
                #print "%s: %s, %s" % (f, y, weight.get())        


class Weight(object):
    def __init__(self):
        self.weight = 0.0
        self.total = 0.0
        self.time = 0

    def get(self):
        return self.weight

    def update(self, value, time):
        """
        update the current weight and summed weight
        :param value:
        :param time:
        :return:
        """
        self.total += (time - self.time) * self.weight
        # update the time of the most recent modification
        self.time = time
        # update weight
        self.weight += value

    def average(self, time):
        self.total += (time - self.time) * self.weight
        self.time = time
        self.weight = self.total / self.time

class AsepectDetector(object):
    """
    determine if a sentence contains aspect
    """
    def __init__(self, name, embeddings=None):
        # classification model
        self.model_name = name
        # parameters used in neural models
        self.embeddings = embeddings

        if name == 'svm':
            self.model = SVC(kernel='linear')
        elif name == 'lr':
            self.model = LogisticRegression()
        elif name == 'rf':
            self.model = RandomForestClassifier(n_estimators=50)
        elif name == 'cnn' or name == 'lstm':
            # build the model at the running time
            pass
            # TODO

    def classify(self, trainset, testset):
        """

        :param trainset:
        :param testset:
        :return:
        """
        train_Y, test_Y = [], []
        train_X, test_X = [], []
        train_words = [sent2words(sent) for sent in trainset]
        test_words = [sent2words(sent) for sent in testset]
        # number of documents
        n_d = len(trainset) + len(testset)
        vocab, df = {}, {}
        dim_w = len(self.embeddings['the'])
        print "dimension of word embedding:", dim_w
        idx = 0
        for doc in train_words + test_words:
            for w in set(doc):
                norm_w = w.lower()
                if norm_w not in vocab:
                    vocab[norm_w] = idx
                    idx += 1
                    df[norm_w] = 1
                else:
                    df[norm_w] += 1
        idf = {}
        for w in df:
            idf[w] = log(1.0 + n_d / float(df[w] + 1.0))

        n_not_find = 0
        for sent in trainset:
            tags = sent2tags(sent)
            words = [w.lower() for w in sent2words(sent)]
            if 'T' in tags:
                train_Y.append(1)
            else:
                train_Y.append(0)
            x = np.zeros(dim_w)
            n_w = 0
            for w in words:
                if w in delset:
                    # ignore the punctuations
                    continue
                if w in self.embeddings:
                    vec = self.embeddings[w]
                else:
                    vec = np.random.uniform(-0.25, 0.25, dim_w)
                    n_not_find += 1
                for i in xrange(len(vec)):
                    #x[i] = x[i] + idf[w] * vec[i]
                    x[i] = x[i] + vec[i]
                n_w += 1
            train_X.append(x / n_w)
        
        for sent in testset:
            tags = sent2tags(sent)
            if 'T' in tags:
                test_Y.append(1)
            else:
                test_Y.append(0)
            words = [w.lower() for w in sent2words(sent)]
            x = np.zeros(dim_w)
            n_w = 0
            filtered_words = []
            for w in words:
                if w in delset:
                    continue
                if w in self.embeddings:
                    vec = self.embeddings[w]
                else:
                    vec = np.random.uniform(-0.25, 0.25, dim_w)
                    n_not_find += 1
                for i in xrange(len(vec)):
                    #x[i] = x[i] + idf[w] * vec[i]
                    x[i] = x[i] + vec[i]
                filtered_words.append(w)
                n_w += 1
            test_X.append(x / n_w)
        print "%s words not exist in word-embeddings..." % n_not_find
        print "Begin training and classification..."
        if self.model_name != 'cnn' and self.model_name != 'lstm' and self.model_name != 'nn':
            # non-neural model
            self.model.fit(train_X, train_Y)
            print "classification results:"
            print self.model.score(test_X, test_Y)
        elif self.model_name == 'nn':
            train_X = np.asarray(train_X)
            test_X = np.asarray(test_X)
            self.model = Sequential()
            self.model.add(Dense(100, input_shape=(dim_w,)))
            self.model.add(Activation('sigmoid'))
            self.model.add(Dense(100))
            self.model.add(Dropout(p=0.5))
            self.model.add(Activation('sigmoid'))
            self.model.add(Dense(1, activation="sigmoid"))

            print "build the neural model..."
            self.model.compile(loss="binary_crossentropy",
                optimizer="adam", metrics=['accuracy'])

            self.model.fit(train_X, train_Y, nb_epoch=30, validation_data=(test_X, test_Y))




