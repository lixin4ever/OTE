__author__ = 'lixin77'

import numpy as np
from tabulate import tabulate
from keras.preprocessing.sequence import pad_sequences
from nltk import ngrams
import os
#from keras.utils.np_utils import to_categorical
from data import Token, Sequence, FeatureIndexer

def word2features(sent, i):
    """
    construct word-level features for each token
    """
    word = sent['words'][i]
    postag = sent['postags'][i]
    #embedding = embeddings[word] if word in embeddings else np.random.uniform(-0.25, 0.25, 200)
    #embedding = [str(ele) for ele in embedding]
    features = {'bias': 1.0, 'word.lower()': word.lower(), 'word.isupper()': word.isupper(),
                'word.isdigit()': word.isdigit(), 'pos_tag': postag, 'prefix1': word[:1] if len(word) >= 1 else '',
                'prefix2': word[:2] if len(word) >= 2 else '', 'prefix3': word[:3] if len(word) >= 3 else '',
                'suffix1': word[-1:] if len(word) >= 1 else '', 'suffix2': word[-2:] if len(word) >= 2 else '',
                'suffix3': word[-3:] if len(word) >= 3 else '',}
    #for idx in xrange(len(embedding)):
    #    features['e%s' % idx] = embedding[idx]
    if i > 0:
        word_l1 = sent['words'][i-1]
        postag_l1 = sent['postags'][i-1]
        features.update({
            '-1:word.lower()': word_l1.lower(),
            '-1:word.istitle()': word_l1.istitle(),
            '-1:word.isupper()': word_l1.isupper(),
            '-1:postag': postag_l1,
        })
    else:
        features['BOS'] = True

    if i < len(sent['words'])-1:
        word_r1 = sent['words'][i+1]
        postag_r1 = sent['words'][i+1]
        features.update({
            '+1:word.lower()': word_r1.lower(),
            '+1:word.istitle()': word_r1.istitle(),
            '+1:word.isupper()': word_r1.isupper(),
            '+1:postag': postag_r1,
        })
    else:
        features['EOS'] = True

    return features


def word2vector(w2v):
    """
    construct features for each word from word embedding
    """
    dim_w = len(w2v)
    features = {}
    for i in xrange(dim_w):
        features['dim%s' % (i + 1)] = w2v[i]
    return features

def sent2features(sent):
    """
    transform sentence to word-level features
    """
    return [word2features(sent, i) for i in xrange(len(sent['words']))]

def sent2embeddings(sent, embeddings):
    """
    transform sentence to embedding-based features
    """
    dim_w = embeddings['the']
    res = []
    if not embeddings:
        # input embeddings are empty
        embeddings = {}
    for w in sent['words']:
        if w.lower() in embeddings:
            res.append(embeddings[w.lower()])
        else:
            res.append(np.random.uniform(-0.25, 0.25, dim_w))
    return res

def sent2tags(sent):
    return [t for t in sent['tags']]

def sent2postags(sent):
    return [t for t in sent['postags']]

def sent2words(sent):
    return [w for w in sent['words']]

def sent2tokens(sent, embeddings=None):
    words = sent2words(sent)
    postags = sent2postags(sent)
    word_vectors = sent2embeddings(sent, embeddings)
    # chunk tags, dependency features or embedding features can also be added
    return [Token(surface=w, pos=postag, embedding=emb) for (w, postag, emb) in zip(words, postags, word_vectors)]

def tag2aspect(tag_sequence):
    """
    convert BIEOS tag sequence to the aspect sequence
    tag_sequence: tag sequence in the BIEOS tagging schema
    """
    n_tag = len(tag_sequence)
    chunk_sequence = []
    beg, end = -1, -1
    # number of multi-word and single-word aspect
    n_mult, n_s = 0, 0
    for i in xrange(n_tag):
        if tag_sequence[i] == 'S':
            # start position and end position are kept same for the singleton
            chunk_sequence.append((i, i))
            n_s += 1
        elif tag_sequence[i] == 'B':
            beg = i
        elif tag_sequence[i] == 'E':
            end = i
            if end > beg:
                # only valid chunk is acceptable
                chunk_sequence.append((beg, end))
                n_mult += 1
    return chunk_sequence, n_s, n_mult

def words2windowFeat(word_seqs, tag_seqs, embeddings):
    """
    generate window-based feature for each word
    :param word_seqs: word sequences
    :param tag_seqs: tag sequences
    :param embeddings: word embedding
    """
    dim_w = len(embeddings['the'])
    X, Y = [], []
    for i in xrange(len(word_seqs)):
        word_seq = word_seqs[i]
        tag_seq = tag_seqs[i]
        assert len(word_seq) == len(tag_seq)
        n_w = len(word_seq)
        for j in xrange(n_w):
            label = int(tag_seq[j] == 'T')
            assert label == 0 or label == 1
            if j == 0:
                features = [ele for ele in embeddings['PADDING']]
            else:
                prev_w = word_seq[j - 1]
                features = [ele for ele in embeddings[prev_w]]
            cur_w = word_seq[j]
            for ele in embeddings[cur_w]:
                features.append(ele)
            if j < n_w - 1:
                next_w = word_seq[j+1]
                for ele in embeddings[next_w]:
                    features.append(ele)
            else:
                for ele in embeddings['PADDING']:
                    features.append(ele)
            assert len(features) == 3 * dim_w
            X.append(features)
            Y.append(label)
    return np.asarray(X), np.asarray(Y)

def label2tag(label_seq, word_seqs):
    """
    transform label sequence to tag sequences for each document
    :param label_seq: 1-d array, containing the output label of model
    :param word_seqs: word sequences
    """
    tag_seqs = []
    # current position of the pointer
    cursor = 0
    # label to tag
    l2t = {0: 'O', 1: 'T'}
    for word_seq in word_seqs:
        n_w = len(word_seq)
        labels = label_seq[cursor:cursor+n_w]
        cursor = cursor + n_w
        tag_seq = [l2t[ele] for ele in labels]
        assert len(tag_seq) == n_w
        tag_seqs.append(tag_seq)
    return tag_seqs


def ot2bieos(tag_sequence):
    """
    convert OT sequence to BIEOS tag sequence
    OT and BIEOS denote tagging schema
    """
    new_sequence = []
    prev = ''
    n_tag = len(tag_sequence)
    for i in xrange(n_tag):
        cur = tag_sequence[i]
        assert cur == 'O' or cur == 'T'
        if cur == 'O':
            new_sequence.append('O')
        else:
            # current tag is T, that is, part of an aspect or a singleton
            if prev != cur:
                # previous tag is not T, current word can only be head word of an aspect or a singleton
                if i == (n_tag - 1):
                    new_sequence.append('S')
                elif tag_sequence[i + 1] == cur:
                    new_sequence.append('B')
                elif tag_sequence[i + 1] != cur:
                    new_sequence.append('S')
                else:
                    raise ValueError('Unexpected tagging case!!')
            else:
                # previous tag is T, current word can only be internal word or the end word of an aspect
                if i == (n_tag - 1):
                    new_sequence.append('E')
                elif tag_sequence[i + 1] == cur:
                    new_sequence.append('I')
                elif tag_sequence[i + 1] != cur:
                    new_sequence.append('E')
                else:
                    raise ValueError('Unexpected tagging case!!')
        prev = cur
    assert len(new_sequence) == len(tag_sequence)
    return new_sequence

def bieos2ot(tag_sequence):
    """
    convert BIEOS sequence to OT tag sequence
    """
    valid_tag = ['B', 'I', 'E', 'O', 'S']
    new_sequence = []
    for ele in tag_sequence:
        assert ele in valid_tag
        if ele == 'O':
            new_sequence.append('O')
        elif ele == 'B':
            new_sequence.append('T')
        elif ele == 'I':
            new_sequence.append('T')
        elif ele == 'E':
            new_sequence.append('T')
        elif ele == 'S':
            new_sequence.append('T')
    # ensure that lengths of two sequences are equal
    assert len(new_sequence) == len(tag_sequence)
    return new_sequence


def evaluate(test_Y, pred_Y):
    """
    evaluate function for sequence tagging task like POS
    """
    assert len(test_Y) == len(pred_Y)
    length = len(test_Y)
    # we just consider T class
    TP, FN, FP = 0, 0, 0
    for i in xrange(length):
        assert len(test_Y[i]) == len(pred_Y[i])
        n_tag = len(test_Y[i])
        for j in xrange(n_tag):
            if test_Y[i][j] == 'T' and pred_Y[i][j] == 'T':
                TP += 1
            if test_Y[i][j] == 'O' and pred_Y[i][j] == 'T':
                FP += 1
            if test_Y[i][j] == 'T' and pred_Y[i][j] == 'O':
                FN += 1
    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    return precision, recall, F1

def evaluate_chunk(test_Y, pred_Y):
    """
    evaluate function for aspect term extraction, generally, it can also be used to evaluate the NER, NP-chunking task
    """
    assert len(test_Y) == len(pred_Y)
    length = len(test_Y)
    TP, FN, FP = 0, 0, 0
    # hit count of mult-word aspect and singleton
    n_mult, n_s = 0, 0
    # gold count of mult-word aspect and singleton
    n_mult_gold, n_s_gold = 0, 0
    # predicted count of mult-word aspect and singleton
    n_mult_pred, n_s_pred = 0, 0
    for i in xrange(length):
        gold = test_Y[i]
        pred = pred_Y[i]
        assert len(gold) == len(pred)
        gold_aspects, n_s_g, n_mult_g = tag2aspect(tag_sequence=ot2bieos(tag_sequence=gold))
        pred_aspects, n_s_p, n_mult_p = tag2aspect(tag_sequence=ot2bieos(tag_sequence=pred))
        n_hit, n_hit_s, n_hit_mult = match_aspect(pred=pred_aspects, gold=gold_aspects)

        n_s += n_hit_s
        n_s_gold += n_s_g
        n_s_pred += n_s_p

        n_mult += n_hit_mult
        n_mult_gold += n_mult_g
        n_mult_pred += n_mult_p

        TP += n_hit
        FP += (len(pred_aspects) - n_hit)
        FN += (len(gold_aspects) - n_hit)
    print tabulate([['singleton', '%s / %s' % (n_s, n_s_gold), '%s / %s' % (n_s, n_s_pred)], ['multi-words', '%s / %s' % (n_mult, n_mult_gold), '%s / %s' % (n_mult, n_mult_pred)], ['total', '%s / %s' % (TP, TP + FN), '%s / %s' % (TP, TP + FP)]], headers={'##', 'recall', 'precision'})
    #print "n_mult:", n_mult
    #print "n_singletor:", n_s
    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    return precision, recall, F1



def match_aspect(pred, gold):
    true_count = 0
    n_mult, n_s = 0, 0
    for t in pred:
        if t in gold:
            true_count += 1
            if t[1] > t[0]:
                n_mult += 1
            else:
                n_s += 1
    return true_count, n_s, n_mult


def to_lower(word_seqs):
    """
    transform all of the word in the lowercase
    """
    res = []
    for word_seq in word_seqs:
        lc_seq = []
        for w in word_seq:
            lc_seq.append(w.lower())
        res.append(lc_seq)
    assert len(res) == len(word_seqs)
    return res

def get_corpus_info(trainset, testset):
    """
    get document frequence, vocabulary from the corpus
    Note: the training set and testing set has been normalized
    """
    df, vocab = {}, {}
    wid = 1 # word id starts from 1
    max_len = -1
    for sent in trainset + testset:
        if max_len < len(sent):
            max_len = len(sent)
        for w in sent:
            if not w in vocab:
                wid += 1
                vocab[w] = wid
                df[w] = 1
            else:
                df[w] += 1
    return vocab, df, max_len

def normalize(word_seq, df):
    """
    normalize the text, mainly for embedding-based extractor
    :param word_seq:
    :param df:
    :return:
    """
    norm_seq = []
    for w in word_seq:
        if w.isdigit():
            norm_seq.append('DIGIT')
        elif df[w] == 1:
            norm_seq.append('UNKNOWN')
        else:
            norm_seq.append(w)
    return norm_seq

def symbol2identifier(X, Y, vocab):
    """
    transform symbol (single word or ngrams) in the dataset to the corresponding ids
    :param X:
    :param Y:
    :param vocab:
    :return:
    """
    id_seqs, label_seqs = [], []
    for symbol_seq in X:
        ids = []
        for s in symbol_seq:
            ids.append(vocab[s])
        id_seqs.append(ids)
    for tag_seq in Y:
        labels = []
        for t in tag_seq:
            labels.append(int(t == 'T'))
        label_seqs.append(labels)
    return id_seqs, label_seqs

def generate_ngram(train, test, n):
    """
    generate ngram sequence
    :param train:
    :param test:
    :param n: Note: n should be an odd number
    :return:
    """
    pad_seq = []
    i = 0
    pad_num = n / 2
    while i < pad_num:
        i += 1
        pad_seq.append('PADDING')
    train_ngrams, test_ngrams = [], []
    for word_seq in train:
        full_seq = pad_seq + word_seq + pad_seq
        n_grams = list(ngrams(full_seq, n))
        assert len(n_grams) == len(word_seq)
        ngram_seq = []
        for t in n_grams:
            for w in list(t):
                ngram_seq.append(w)
        assert len(ngram_seq) == n * len(word_seq)
        train_ngrams.append(ngram_seq)
    assert len(train_ngrams) == len(train)
    for word_seq in test:
        full_seq = pad_seq + word_seq + pad_seq
        n_grams = list(ngrams(full_seq, n))
        assert len(n_grams) == len(word_seq)
        ngram_seq = []
        for t in n_grams:
            for w in list(t):
                ngram_seq.append(w)
        assert len(ngram_seq) == n * len(word_seq)
        test_ngrams.append(ngram_seq)
    assert len(test_ngrams) == len(test)
    return train_ngrams, test_ngrams


def padding_zero(X, Y, max_len, winsize=1):
    """
    padding the word sequences and tag (label) sequences
    :param X: input word sequences
    :param Y: label sequences of the corresponding words
    :param max_len: maximum length of the sequence
    :param winsize: window size of the window-based word representation
    :return:
    """
    padded_X, padded_Y = pad_sequences(X, maxlen=max_len * winsize, padding='post'), pad_sequences(Y, maxlen=max_len, padding='post')
    padded_Y = np.reshape(padded_Y, (padded_Y.shape[0], padded_Y.shape[1], 1))
    return padded_X, padded_Y

def padding_special(X, Y, max_len, winsize, special_value):
    """
    add (winsize - 1) special token for each sequence and then perform padding
    :param X:
    :param Y:
    :param max_len:
    :param winsize: window size
    :param special_value: added value before padding, which is a required argument
    :return:
    """
    padded_seq = []
    n_pad, i = winsize / 2, 0
    while i < n_pad:
        i += 1
        padded_seq.append(special_value)
    new_X, new_test = [], []
    for seq in X:
        new_seq = padded_seq + seq + padded_seq
        new_X.append(new_seq)
    padded_X, padded_Y = pad_sequences(X, maxlen=max_len + winsize - 1, padding='post'), pad_sequences(Y, maxlen=max_len, padding='post')
    padded_Y = np.reshape(padded_Y, (padded_Y.shape[0], padded_Y.shape[1], 1))
    return padded_X, padded_Y

def get_valid_seq(padded_seq, raw_len):
    """
    get valid tag sequence from the predicted, padded sequence
    :param raw_len: original length of the corresponding sequence
    :param padded_seq: padded sequence predicted from raw length
    :return:
    """
    raw_seq = []
    identifier2tag = {0: 'O', 1: 'T'}
    for i in xrange(raw_len):
        raw_seq.append(identifier2tag[padded_seq[i]])
    return raw_seq

def output(test_set, pred_Y, model_name):
    """
    write the output back to the disk
    :param test_set:
    :param pred_Y:
    :param model_name:
    :return:
    """
    assert len(test_set) == len(pred_Y)
    n_sen = len(test_set)
    lines = []
    for i in xrange(n_sen):
        tokens = sent2tokens(test_set[i])
        pred = pred_Y[i]
        assert len(tokens) == len(pred)
        aspects, _, _ = tag2aspect(tag_sequence=ot2bieos(tag_sequence=pred))
        sent = ' '.join(tokens)
        aspect_terms = []
        for (b, e) in aspects:
            at = '_'.join(tokens[b: (e+1)])
            aspect_terms.append(at)
        lines.append('%s\n' % ('##'.join([sent] + aspect_terms)))
    if not os.path.exists('./res'):
        os.mkdir('./res')
    with open('./res/%s.txt' % model_name, 'w+') as fp:
        fp.writelines(lines)

def preprocess_seq(x):
    """
    collect features for each token in sequence x
    :param x: a sequence of Token object
    :return:
    """
    x[0].add(['word.BOS=True'])
    x[-1].add(['word.EOS=True'])
    for t in x:
        assert isinstance(t, Token)
        t.add(token_features(token=t))
    # feature from x[t-1]
    for i in xrange(1, len(x)):
        x[i].add(f + '@-1' for f in token_features(x[i-1]))
    # feature from x[t+1]
    for i in xrange(0, len(x) - 1):
        x[i].add(f + '@+1' for f in token_features(x[i+1]))
    return x

def token_features(token):
    """
    generate observation features, follow the idea in https://github.com/ppfliu/opinion-target/blob/master/absa.py
    """
    w = token.surface
    embedding = token.embedding
    # word identity
    yield 'word=%s' % w
    w_shape = get_shape(w)
    yield 'word.shape=%s' % w_shape
    # lower- or upper-case
    yield 'word.lower=%s' % w.lower()
    yield 'word.upper=%s' % w.upper()

    # is-xxx feature
    yield 'word.isdigit=%s' % w.isdigit()
    yield 'word.isupper=%s' % w.isupper()
    yield 'word.isalpha=%s' % w.isalpha()
    yield 'word.istitle=%s' % w.istitle()

    # prefix and suffix
    if len(w) >= 3:
        p3 = w[:3]
        s3 = w[-3:]
    else:
        p3 = ''
        s3 = ''
    if len(w) >= 2:
        p2 = w[:2]
        s2 = w[-2:]
    else:
        p2 = ''
        s2 = ''
    if len(w) >= 1:
        p1 = w[:1]
        s1 = w[-1:]
    else:
        p1 = ''
        s1 = ''
    yield 'word.prefix3=%s' % p3
    yield 'word.suffix3=%s' % s3
    yield 'word.prefix2=%s' % p2
    yield 'word.suffix2=%s' % s2
    yield 'word.prefix1=%s' % p1
    yield 'word.suffix1=%s' % s1
    if embedding:
        for i in xrange(len(embedding)):
            yield 'embedding.%s=%s' % (i, embedding[i])

def get_shape(word):
    s = ''
    for ch in word:
        if ch.isupper():
            s += 'U'
        elif ch.islower():
            s += 'L'
        elif ch.isdigit():
            s += 'D'
        elif ch in ('.', ','):
            s += '.'
        elif ch in (';', ':', '?', '!'):
            s += ';'
        elif ch in ('+', '-', '*', '/', '=', '|', '_'):
            s += '-'
        elif ch in ('(', '<', '{', '['):
            s += '('
        elif ch in (')', '>', '}', ']'):
            s += ')'
        else:
            s += ch
    return ch

def build_indexer(training_data):
    """

    :param training_data:
    :return:
    """
    # state / label feature indexer
    label_indexer = FeatureIndexer()
    # observation / token feature indexer
    observation_indexer = FeatureIndexer()
    for x in training_data:
        assert isinstance(x, Sequence)
        tags = x.tags
        f = x.F
        # first token
        observation_indexer.add_many(f(t=0, yt_1=None, yt=tags[0]))
        for t in xrange(1, len(x)):
            observation_indexer.add_many(f(t=t, yt_1=tags[t-1], yt=tags[t]))
        label_indexer.add_many(tags)
    observation_indexer._frozen()
    print "number of feature functions:", len(observation_indexer)
    return observation_indexer, label_indexer

def to_array(features, feat_vocab, dtype='float64'):
    """
    transform dictionary into array
    :param features: feature dictionary
    :param feat_vocab: mapping between feature and feature id
    :param dtype:
    :return:
    """
    n_feature = len(feat_vocab)
    X_array = []
    for seq in features:
        x = []
        for feat_map in seq:
            feature_array = np.zeros(n_feature)
            for k in feat_map:
                fid = feat_vocab[k]
                v = feat_map[k]
                if v == 'True':
                    v = 1
                if v == 'False':
                    v = 0
                feature_array[fid] = v
            x.append(feature_array.astype(dtype=dtype))
        X_array.append(x)
    return X_array

def feature_extractor(data, _type='map', feat='word', embeddings=None):
    """
    feature extractor for crfsuite or other existing taggers
    :param data:
    :param _type: input form of features, for crfsuite, it is "map", for the svm and other learning, it is "array"
    :param feat: type of features, word-level features and embedding features are available
    :param embeddings: pre-trained or random-initialized word embeddings
    :return:
    """
    X = []
    indexer = FeatureIndexer()
    for sent in data:
        seq_features = []
        tokens = sent2tokens(sent, embeddings)
        for i in xrange(len(tokens)):
            tmp = {'bias': 1.0}
            assert isinstance(tokens[i], Token)
            for f in tokens[i].observations:
                k, v = f.split('=')
                if k.startswith('word') and feat == 'embedding':
                    continue
                if k.startswith('embedding') and feat == 'word':
                    continue
                tmp[k] = v
                indexer.add(k)
            if i > 0:
                for f in tokens[i-1].observations:
                    k, v = f.split('=')
                    if k.startswith('word') and feat == 'embedding':
                        continue
                    if k.startswith('embedding') and feat == 'word':
                        continue
                    cur_k = '%s@-1' % k
                    tmp[cur_k] = v
                    indexer.add(cur_k)
            else:
                if feat == 'word':
                    # using word-level features
                    tmp['word.BOS'] = "True"
                    indexer.add('word.BOS')
            if i < len(tokens) - 1:
                for f in tokens[i+1].observations:
                    k, v = f.split('=')
                    if k.startswith('word') and feat == 'embedding':
                        continue
                    if k.startswith('embedding') and feat == 'word':
                        continue
                    cur_k = '%s@+1' % k
                    tmp[cur_k] = v
                    indexer.add(cur_k)
            else:
                if feat == 'word':
                    tmp['word.EOS'] = "True"
            seq_features.append(tmp)
        X.append(seq_features)
    if _type == 'string':
        return X
    else:
        return to_array(features=X, feat_vocab=indexer._mapping)

