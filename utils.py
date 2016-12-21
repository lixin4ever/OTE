__author__ = 'lixin77'

import numpy as np
from tabulate import tabulate
from keras.preprocessing.sequence import pad_sequences

def contain_upper(word):
    for ch in word:
        if 'A' <= ch <= 'Z':
            return True
    return False

def contain_lower(word):
    for ch in word:
        if 'a' <= ch <= 'z':
            return True
    return False

def contain_digit(word):
    for ch in word:
        if '0' <= ch <= "9":
            return True
    return False

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
    return [word2vector(w2v=embeddings[w]) for w in sent]

def sent2tags(sent):
    return [t for t in sent['tags']]

def sent2postags(sent):
    return [t for t in sent['postags']]

def sent2tokens(sent):
    return [w for w in sent['words']]



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
                features = [ele for ele in np.random.uniform(-0.25, 0.25, dim_w)]
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
                for ele in np.random.uniform(-0.25, 0.25, dim_w):
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
    evaluate function for aspect term extraction, generally, it can also be used to evaluate the NER, chunking task
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

def token2identifier(X, Y, vocab):
    """
    transform words in the dataset to the word ids
    :param dataset:
    :param vocab:
    :return:
    """
    wid_seqs, label_seqs = [], []
    for word_seq in X:
        wids = []
        for w in word_seq:
            wids.append(vocab[w])
        wid_seqs.append(wids)
    for tag_seq in Y:
        labels = []
        for t in tag_seq:
            labels.append(int(t == 'T'))
        label_seqs.append(labels)
    return wid_seqs, label_seqs

def padding(X, Y, max_len):
    """
    padding the word sequences and tag (label) sequences
    :param X: input word sequences
    :param Y: label sequences of the corresponding words
    :param max_len: maximum length of the sequence
    :return:
    """
    return pad_sequences(X, maxlen=max_len), pad_sequences(Y, maxlen=max_len)

def get_valid_seq(padded_seq, raw_len):
    """
    get valid tag sequence from the predicted padded sequence
    :param raw_len: original length of the corresponding sequence
    :param padded_seq: padded sequence predicted from raw length
    :return:
    """
    raw_seq = []
    identifier2tag = {0: 'O', 1: 'T'}
    for i in xrange(raw_len):
        raw_seq.append(identifier2tag[padded_seq[i]])
    return raw_seq
