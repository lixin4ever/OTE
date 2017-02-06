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
from keras.layers import Conv1D, LSTM, Dense, Embedding, Dropout, Activation, Input, GlobalMaxPooling1D, merge
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.constraints import maxnorm
from nltk.corpus import stopwords
import sklearn_crfsuite
from sklearn_crfsuite import metrics


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
        dim_symbol = len(self.embeddings['the'])
        print "dimension of word embedding:", dim_symbol
        if self.model_name == 'lstm' or self.model_name == 'cnn':
            # idx starts from 1 because zero padding is needed in the neural model
            idx = 1
        else:
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
            x = np.zeros(dim_symbol)
            n_w = 0
            for w in words:
                if w in delset:
                    # ignore the punctuations
                    continue
                if w in self.embeddings:
                    vec = self.embeddings[w]
                else:
                    vec = np.random.uniform(-0.25, 0.25, dim_symbol)
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
            x = np.zeros(dim_symbol)
            n_w = 0
            filtered_words = []
            for w in words:
                if w in delset:
                    continue
                if w in self.embeddings:
                    vec = self.embeddings[w]
                else:
                    vec = np.random.uniform(-0.25, 0.25, dim_symbol)
                    n_not_find += 1
                for i in xrange(len(vec)):
                    #x[i] = x[i] + idf[w] * vec[i]
                    x[i] = x[i] + vec[i]
                filtered_words.append(w)
                n_w += 1
            test_X.append(x / n_w)
        print "Begin training and classification..."
        if self.model_name != 'cnn' and self.model_name != 'lstm' and self.model_name != 'nn':
            # non-neural model
            self.model.fit(train_X, train_Y)
            print "classification results:"
            #print self.model.score(test_X, test_Y)
            pred_Y = self.model.predict(test_X)
        elif self.model_name == 'nn':
            train_X = np.asarray(train_X)
            test_X = np.asarray(test_X)
            self.model = Sequential()
            self.model.add(Dense(100, input_shape=(dim_symbol,)))
            self.model.add(Activation('sigmoid'))
            self.model.add(Dense(100))
            self.model.add(Dropout(p=0.5))
            self.model.add(Activation('sigmoid'))
            self.model.add(Dense(1, activation="sigmoid"))

            print "build the FCN..."
            self.model.compile(loss="binary_crossentropy",
                optimizer="adam", metrics=['accuracy'])

            self.model.fit(train_X, train_Y, nb_epoch=30, validation_data=(test_X, test_Y))
            pred_Y = []
        elif self.model_name == 'lstm' or self.model_name == 'cnn':
            n_symbole = len(vocab)
            train_X, test_X = [], []
            embedding_weights = np.zeros((n_symbole + 1, dim_symbol))
            for w in vocab:
                idx = vocab[w]
                if w in self.embeddings:
                    embedding_weights[idx, :] = self.embeddings[w]
                else:
                    embedding_weights[idx, :] = np.random.uniform(-0.25, 0.25, dim_symbol)
            max_len = 0
            for words in train_words:
                ids = []
                for w in words:
                    norm_w = w.lower()
                    ids.append(vocab[norm_w])
                if max_len < len(words):
                    max_len = len(words)
                train_X.append(ids)
            for words in test_words:
                ids = []
                for w in words:
                    norm_w = w.lower()
                    ids.append(vocab[norm_w])
                if max_len < len(words):
                    max_len = len(words)
                test_X.append(ids)
            train_X = pad_sequences(sequences=train_X, maxlen=max_len)
            test_X = pad_sequences(sequences=test_X, maxlen=max_len)

            model_input = Input(shape=(max_len, ), name='%s_input' % self.model_name)
            # obtain sentence embeddings
            if self.model_name == 'lstm':
                mask_zero = True
            else:
                mask_zero = False
            sent_embeddings = Embedding(input_dim=n_symbole + 1, output_dim=dim_symbol,
                                   input_length=max_len, weights=[embedding_weights],
                                    dropout=0.2, mask_zero=mask_zero)(model_input)
            if self.model_name == 'lstm':
                print "build the lstm model..."
                optimizer = 'adam'
                lstm_out = LSTM(output_dim=128, dropout_U=0.2, dropout_W=0.2)(sent_embeddings)
                model_output = Dense(1, activation='sigmoid', name='model_output')(lstm_out)
            if self.model_name == 'cnn':
                # use cnn model in Kim et al (EMNLP 2014)
                optimizer = 'adadelta'
                print "build the cnn model..."
                nb_filter = 100

                para_constrains = maxnorm(m=3)
                conv1_output = Conv1D(nb_filter=nb_filter, filter_length=3, activation='relu',
                                      border_mode='valid', b_constraint=para_constrains,
                                      W_constraint=para_constrains)(sent_embeddings)
                conv_pool1_output = GlobalMaxPooling1D()(conv1_output)

                conv2_output = Conv1D(nb_filter=nb_filter, filter_length=4, activation='relu',
                                      border_mode='valid', b_constraint=para_constrains,
                                      W_constraint=para_constrains)(sent_embeddings)
                conv_pool2_output = GlobalMaxPooling1D()(conv2_output)

                conv3_output = Conv1D(nb_filter=nb_filter, filter_length=5, activation='relu',
                                      border_mode='valid', b_constraint=para_constrains,
                                      W_constraint=para_constrains)(sent_embeddings)
                conv_pool3_output = GlobalMaxPooling1D()(conv3_output)

                x = merge(inputs=[conv_pool1_output, conv_pool2_output, conv_pool3_output], mode='concat')

                x_dropout = Dropout(p=0.5)(x)

                model_output = Dense(1, activation='sigmoid', name='model_output')(x_dropout)
            self.model = Model(input=[model_input], output=[model_output])

            self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


            best_model = ModelCheckpoint(filepath='./model/%s.hdf5' % self.model_name,
                                               monitor='val_acc', save_best_only=True, mode='max')
            self.model.fit(train_X, train_Y, nb_epoch=50,
                           validation_data=(test_X, test_Y),
                           callbacks=[best_model])
            # load best model from the disk
            self.model.load_weights(filepath='./model/%s.hdf5' % self.model_name)
            """
            self.model = Sequential()
            self.model.add(Embedding(input_dim=n_symbole + 1, output_dim=dim_symbol,
                                     input_length=max_len, weights=[embedding_weights],
                                     dropout=0.2, mask_zero=True))
            self.model.add(LSTM(output_dim=128, dropout_U=0.2, dropout_W=0.2))
            self.model.add(Dense(1, activation='sigmoid'))

            print "build the lstm..."
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


            self.model.fit(train_X, train_Y, nb_epoch=100, validation_data=(test_X, test_Y),
                           callbacks=[model_checkpoint])

            """
            pred_Y = self.model.predict(test_X)
        return pred_Y


class Segment(object):
    def __init__(self, beg, end, aspect):
        """
        :param beg: begin position
        :param end: end position
        """
        self.beg = beg
        self.end = end
        self.aspect = aspect


class HierachyExtractor(object):
    """
    segmentation-based extractor
    """
    def __init__(self, seg_name, labeler_name):
        if seg_name == 'crf':
            self.segmentor = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True)
        else:
            self.segmentor = None
        if labeler_name == 'crf':
            self.labeler = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True)
        else:
            self.labeler = None


    def fit(self, dataset, embeddings):
        """
        train segmentor and labeler
        :param dataset: training set
        :return:
        """
        from utils import sent2features, aspect2segment, tag2seg, sent2tags_seg, sent2seg_features, sent2tags
        # not using embeddings
        train_X = [sent2features(sent, embeddings) for sent in dataset]
        # segment tag sequence, not original aspect tag sequence
        train_Y = [aspect2segment(sent2tags(sent)) for sent in dataset]
        print "train the segmentor..."
        self.segmentor.fit(train_X, train_Y)
        train_X_seg = []
        train_Y_seg = []
        for (i, tag_seq) in enumerate(train_Y):
            sent = dataset[i]
            segments = tag2seg(tag_sequence=tag_seq, sent=sent)
            train_X_seg.append(sent2seg_features(segments=segments, sent=sent, embeddings=embeddings))
            train_Y_seg.append(sent2tags_seg(sent=segments))

        print "train the labeler..."
        self.labeler.fit(train_X_seg, train_Y_seg)


    def predict(self, dataset, embeddings, ds_name):
        """
        predict segment boundary and label
        :param dataset: testing dataset
        :return:
        """
        from utils import sent2features, tag2seg, sent2seg_features, sent2tags_seg, \
            sent2tags, segment2aspect, evaluate_chunk, aspect2segment
        # input of segmentor
        test_X = [sent2features(sent, embeddings) for sent in dataset]
        test_Y = [sent2tags(sent) for sent in dataset]
        # predict segment boundaries
        pred_seqs = self.segmentation(X=test_X)
        # gold segment boundaries
        gold_seqs = [aspect2segment(aspect_sequence=seq) for seq in test_Y]
        print evaluate_chunk(test_Y=gold_seqs, pred_Y=pred_seqs, testset=dataset, model_name='hCRF', ds_name=ds_name)
        # input of labeler
        test_X_seg = []
        pred_segments = []
        for i in xrange(len(pred_seqs)):
            sent = dataset[i]
            # predicted sentence boundaries
            pred_seq = pred_seqs[i]
            # predicted segments
            segments = tag2seg(tag_sequence=pred_seq, sent=sent)
            test_X_seg.append(sent2seg_features(segments=segments, sent=sent, embeddings=embeddings))
            pred_segments.append(segments)
        # predicted aspect tags
        pred_Y_seg = self.label(X=test_X_seg)

        assert len(pred_Y_seg) == len(pred_segments)
        pred_Y = []
        for (i, tag_seq) in enumerate(pred_Y_seg):
            pred_Y.append(segment2aspect(tags=tag_seq, segments=pred_segments[i]))
        print evaluate_chunk(test_Y=test_Y, pred_Y=pred_Y, testset=dataset, model_name='hCRF', ds_name=ds_name)

    def segmentation(self, X):
        return self.segmentor.predict(X)

    def label(self, X):
        return self.labeler.predict(X)




