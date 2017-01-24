__author__ = 'lixin77'
import cPickle
import sklearn_crfsuite
from sklearn_crfsuite import metrics, scorers
from sklearn.svm import SVC
import sys
from utils import *
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Embedding, TimeDistributed, Dense, Reshape, Convolution1D
from keras.callbacks import EarlyStopping
from models import AveragedPerceptron, AsepectDetector
from numpy import linalg as LA
#from layers import MyEmbedding

#embeddings = {}
def crf_extractor(train_set, test_set, embeddings=None, model_name=None, ds_name=None):
    """
    linear-chain crf extractor with basic feature template
    :param train_set: training dataset
    :param test_set: testing dataset
    """
    train_Y = [sent2tags(sent) for sent in train_set]
    test_Y = [sent2tags(sent) for sent in test_set]

    n_nsubj_train = sum([1 for tags in train_Y if 'T' not in tags])
    n_nsubj_test = sum([1 for tags in test_Y if 'T' not in tags])
    print "n_train_w/o_aspect:", n_nsubj_train
    print "n_test_w/o_aspect:", n_nsubj_test
    #AD = AsepectDetector(name='cnn', embeddings=embeddings)
    #embeddings = None
    #pred_res = AD.classify(trainset=train_set, testset=test_set)
    # filtered sentence without aspects
    #new_test_set = [test_set[i] for (i, y) in enumerate(pred_res) if y > 0.5]
    new_test_set = test_set
    if True:
        print "crf with word-level features..."
        #train_X = feature_extractor(data=train_set, _type='map', feat='word', embeddings=embeddings)
        #test_X = feature_extractor(data=test_set, _type='map', feat='word', embeddings=embeddings)
        train_X = [sent2features(sent, embeddings) for sent in train_set]
        test_X = [sent2features(sent, embeddings) for sent in new_test_set]
        test_Y = [sent2tags(sent) for sent in new_test_set]
    else:
        print "crf with word embeddings..."
        train_words = [sent2words(sent) for sent in train_set]
        #train_Y = [sent2tags(sent) for sent in train_set]
        train_words = to_lower(word_seqs=train_words)

        test_words = [sent2words(sent) for sent in test_set]
        test_words = to_lower(word_seqs=test_words)
        #test_Y = [sent2tags(sent) for sent in test_set]

        vocab, df, max_len = get_corpus_info(trainset=train_words, testset=test_words)

        # replace words appearing infrequently with "UNKNOWN"
        train_words_norm = [normalize(seq, df) for seq in train_words]
        test_words_norm = [normalize(seq, df) for seq in test_words]

        dim_w = len(embeddings['the'])
        embeddings['DIGIT'] = np.random.uniform(-0.25, 0.25, dim_w)
        embeddings['UNKNOWN'] = np.random.uniform(-0.25, 0.25, dim_w)

        train_X = [sent2embeddings(sent, embeddings) for sent in train_words_norm]
        test_X = [sent2embeddings(sent, embeddings) for sent in test_words_norm]


    crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True)

    print "begin crf training..."
    crf.fit(train_X, train_Y)

    pred_Y = crf.predict(test_X)

    print "shape of output: (%s, %s)" % (len(pred_Y), len(pred_Y[0]))

    labels = list(crf.classes_)
    labels.remove('O')

    print labels

    print(metrics.flat_classification_report(test_Y, pred_Y, labels=labels, digits=3))

    print evaluate_chunk(test_Y=test_Y, pred_Y=pred_Y, testset=new_test_set, 
        model_name=model_name, ds_name=ds_name)

    output(test_set=new_test_set, pred_Y=pred_Y, model_name=model_name)

def svm_extractor(train_set, test_set, embeddings=None):
    """
    window-based support vector machine for aspect extraction
    :param train_set: training set
    :param test_set: testing set
    :param embeddings: pretrained word_embeddings, NONE means does not use word embeddings
    """
    train_words = [sent2tokens(sent) for sent in train_set]
    train_tags = [sent2tags(sent) for sent in train_set]
    train_words = to_lower(word_seqs=train_words)

    test_words = [sent2tokens(sent) for sent in test_set]
    test_words = to_lower(word_seqs=test_words)
    test_tags = [sent2tags(sent) for sent in test_set]

    vocab, df, max_len = get_corpus_info(trainset=train_words, testset=test_words)

    train_words_norm = [normalize(seq, df) for seq in train_words]
    test_words_norm = [normalize(seq, df) for seq in test_words]

    dim_w = len(embeddings['the'])
    embeddings['DIGIT'] = np.random.uniform(-0.25, 0.25, dim_w)
    embeddings['UNKNOWN'] = np.random.uniform(-0.25, 0.25, dim_w)
    embeddings['PADDING'] = np.random.uniform(-0.25, 0.25, dim_w)

    train_X, train_Y = words2windowFeat(word_seqs=train_words_norm, tag_seqs=train_tags, embeddings=embeddings)

    test_X, test_Y = words2windowFeat(word_seqs=test_words_norm, tag_seqs=test_tags, embeddings=embeddings)

    clf = SVC(kernel='linear')
    print "begin svm training..."
    clf.fit(train_X, train_Y)
    pred_Y = clf.predict(test_X)
    assert len(pred_Y) == len(test_Y)
    pred_tags = label2tag(label_seq=pred_Y, word_seqs=test_words)
    print evaluate_chunk(test_Y=test_tags, pred_Y=pred_tags)

def lstm_extractor(train_set, test_set, embeddings, win_size=1):
    """
    LSTM extractor for aspect term extraction, text pre-processing step follows the
    paper: http://www.aclweb.org/anthology/D/D15/D15-1168.pdf
    :param train_set:
    :param test_set:
    :param embeddings:
    :param win_size: window size, should be an odd number
    """
    print "Bi-directional LSTM with word embeddings..."
    train_words = [sent2words(sent) for sent in train_set]
    train_tags = [sent2tags(sent) for sent in train_set]
    train_words = to_lower(word_seqs=train_words)

    test_words = [sent2words(sent) for sent in test_set]
    test_words = to_lower(word_seqs=test_words)
    test_tags = [sent2tags(sent) for sent in test_set]

    vocab, df, max_len = get_corpus_info(trainset=train_words, testset=test_words)

    train_words_norm = [normalize(seq, df) for seq in train_words]
    test_words_norm = [normalize(seq, df) for seq in test_words]

    dim_w = len(embeddings['the'])
    embeddings['DIGIT'] = np.random.uniform(-0.25, 0.25, dim_w)
    embeddings['UNKNOWN'] = np.random.uniform(-0.25, 0.25, dim_w)
    embeddings['PADDING'] = np.random.uniform(-0.25, 0.25, dim_w)
    n_w = len(vocab)
    #vocab['PADDING'] = 0
    if not 'DIGIT' in vocab:
        vocab['DIGIT'] = n_w + 1
        n_w += 1
    if not 'UNKNOWN' in vocab:
        vocab['UNKNOWN'] = n_w + 1
        n_w += 1
    if not 'PADDING' in vocab:
        vocab['PADDING'] = n_w + 1
        n_w += 1


    embeddings_unigram = np.zeros((n_w + 1, dim_w))
    for (w, idx) in vocab.items():
        embeddings_unigram[idx, :] = embeddings[w]
    embeddings_weights = embeddings_unigram

    if win_size == 1:
        train_X, train_Y = symbol2identifier(X=train_words_norm, Y=train_tags, vocab=vocab)
        test_X, test_Y = symbol2identifier(X=test_words_norm, Y=test_tags, vocab=vocab)
        # padding the symbol sequence
        train_X, train_Y = padding_zero(train_X, train_Y, max_len=max_len)
        test_X, test_Y = padding_zero(test_X, test_Y, max_len=max_len)
    else:
        train_ngrams, test_ngrams = generate_ngram(train=train_words_norm, test=test_words_norm, n=win_size)
        train_X, train_Y = symbol2identifier(X=train_ngrams, Y=train_tags, vocab=vocab)
        test_X, test_Y = symbol2identifier(X=test_ngrams, Y=test_tags, vocab=vocab)
        # padding the symbol sequence
        train_X, train_Y = padding_zero(train_X, train_Y, max_len=max_len, winsize=win_size)
        test_X, test_Y = padding_zero(test_X, test_Y, max_len=max_len, winsize=win_size)
    n_symbol = n_w
    dim_symbol = dim_w

    print "train shape:", train_X.shape
    print "test shape:", test_X.shape

    print "Build the Bi-LSTM model..."
    LSTM_extractor = Sequential()
    LSTM_extractor.add(Embedding(output_dim=dim_symbol,
                                   input_dim=n_symbol + 1, weights=[embeddings_weights],
                                   mask_zero=False, input_length=max_len*win_size))
    LSTM_extractor.add(Reshape((max_len, win_size * dim_symbol)))
    LSTM_extractor.add(Bidirectional(LSTM(100, return_sequences=True), merge_mode='concat'))
    #LSTM_extractor.add(LSTM(100, return_sequences=True))
    LSTM_extractor.add(TimeDistributed(Dense(output_dim=1, activation='sigmoid')))
    #LSTM_extractor.add(Flatten())
    LSTM_extractor.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print LSTM_extractor.summary()
    print "Begin to training the model..."
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    LSTM_extractor.fit(train_X, train_Y, batch_size=32, nb_epoch=30,
                       validation_data=(test_X, test_Y), callbacks=[early_stopping])

    res = LSTM_extractor.predict_classes(test_X)
    res = res.reshape((res.shape[0], res.shape[1] * res.shape[2]))

    print "output dim:", res.shape
    #assert res.shape == test_X.shape
    assert res.shape[0] == len(test_X)
    pred_tags = []
    for (i, raw_seq) in enumerate(test_tags):
        pred_tags.append(get_valid_seq(padded_seq=res[i], raw_len=len(raw_seq)))

    print evaluate_chunk(test_Y=test_tags, pred_Y=pred_tags, testset=test_set)

def conv_lstm_extractor(train_set, test_set, embeddings, win_size):
    """

    :param train_set:
    :param test_set:
    :param embeddings:
    :param win_size: height of convolutional filter
    :return:
    """
    print "Convolutional Bi-direction LSTM with word embeddings..."
    train_words = [sent2words(sent) for sent in train_set]
    train_tags = [sent2tags(sent) for sent in train_set]
    train_words = to_lower(word_seqs=train_words)

    test_words = [sent2words(sent) for sent in test_set]
    test_words = to_lower(word_seqs=test_words)
    test_tags = [sent2tags(sent) for sent in test_set]

    vocab, df, max_len = get_corpus_info(trainset=train_words, testset=test_words)

    train_words_norm = [normalize(seq, df) for seq in train_words]
    test_words_norm = [normalize(seq, df) for seq in test_words]

    dim_w = len(embeddings['the'])
    embeddings['DIGIT'] = np.random.uniform(-0.25, 0.25, dim_w)
    embeddings['UNKNOWN'] = np.random.uniform(-0.25, 0.25, dim_w)
    embeddings['PADDING'] = np.random.uniform(-0.25, 0.25, dim_w)
    n_w = len(vocab)
    #vocab['PADDING'] = 0
    if not 'DIGIT' in vocab:
        vocab['DIGIT'] = n_w + 1
        n_w += 1
    if not 'UNKNOWN' in vocab:
        vocab['UNKNOWN'] = n_w + 1
        n_w += 1
    if not 'PADDING' in vocab:
        vocab['PADDING'] = n_w + 1
        n_w += 1
    n_w = len(vocab)
    #vocab['PADDING'] = 0
    if not 'DIGIT' in vocab:
        vocab['DIGIT'] = n_w + 1
        n_w += 1
    if not 'UNKNOWN' in vocab:
        vocab['UNKNOWN'] = n_w + 1
        n_w += 1
    if not 'PADDING' in vocab:
        vocab['PADDING'] = n_w + 1
        n_w += 1

    train_X, train_Y = symbol2identifier(X=train_words_norm, Y=train_tags, vocab=vocab)
    test_X, test_Y = symbol2identifier(X=test_words_norm, Y=test_tags, vocab=vocab)

    train_X, train_Y = padding_special(X=train_X, Y=train_Y, max_len=max_len, winsize=win_size, special_value=vocab['PADDING'])
    test_X, test_Y = padding_special(X=test_X, Y=test_Y, max_len=max_len, winsize=win_size, special_value=vocab['PADDING'])

    print "train_X: %s, train_Y:%s" % (train_X.shape, train_Y.shape)
    print "test_X: %s, test_Y:%s" % (test_X.shape, test_Y.shape)

    n_symbol = n_w
    dim_symbol = dim_w

    embeddings_unigram = np.zeros((n_w + 1, dim_w))
    for (w, idx) in vocab.items():
        embeddings_unigram[idx, :] = embeddings[w]
    embeddings_weights = embeddings_unigram

    ConvLSTM_extractor = Sequential()

    ConvLSTM_extractor.add(Embedding(output_dim=dim_symbol,
                                   input_dim=n_symbol + 1, weights=[embeddings_weights],
                                   mask_zero=False, input_length=max_len+win_size-1))
    ConvLSTM_extractor.add(Convolution1D(nb_filter=1, filter_length=win_size,
                                            input_shape=(max_len+win_size-1, dim_symbol)))
    ConvLSTM_extractor.add(Bidirectional(LSTM(100, return_sequences=True), merge_mode='concat'))
    ConvLSTM_extractor.add(TimeDistributed(Dense(output_dim=1, activation='sigmoid')))
    ConvLSTM_extractor.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print ConvLSTM_extractor.summary()

    print "Begin to train the model..."
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    ConvLSTM_extractor.fit(train_X, train_Y, batch_size=32, nb_epoch=30,
                       validation_data=(test_X, test_Y), callbacks=[early_stopping])

    res = ConvLSTM_extractor.predict_classes(test_X)
    res = res.reshape((res.shape[0], res.shape[1] * res.shape[2]))

    assert res.shape[0] == len(test_X)
    print "output dim:", res.shape
    pred_tags = []
    for (i, raw_seq) in enumerate(test_tags):
        pred_tags.append(get_valid_seq(padded_seq=res[i], raw_len=len(raw_seq)))

    print evaluate_chunk(test_Y=test_tags, pred_Y=pred_tags)

    output(test_set=test_set, pred_Y=pred_tags, model_name='ConvLSTM')

def ap_extractor(train_set, test_set):
    """
    averaged perceptron extractor
    :param train_set: training set
    :param test_set: testing set
    :return:
    """
    ap = AveragedPerceptron()
    ap.fit(trainset=train_set)
    Y_pred = ap.infer(testset=test_set)
    Y_gold = [sent2tags(sent) for sent in test_set]
    print evaluate_chunk(test_Y=Y_gold, pred_Y=Y_pred)

    output(test_set=test_set, pred_Y=Y_pred, model_name='ap')


def run(ds_name, model_name='crf', feat='word'):
    """

    :param ds_name: dataset name
    :param model_name: model name
    :param feat: features used in the learning process
    """
    print "load dataset from disk..."
    train_set = cPickle.load(open('./pkl/%s_train.pkl' % ds_name, 'rb'))
    test_set = cPickle.load(open('./pkl/%s_test.pkl' % ds_name, 'rb'))

    glove_embeddings, embeddings = {}, {}
    embedding_path = '../yelp/yelp_vec_200_2.txt'
    print "load word embeddings from %s..." % embedding_path
    with open(embedding_path, 'r') as fp:
        for line in fp:
            values = line.strip().split()
            word, vec = values[0], values[1:]
            glove_embeddings[word] = vec
    n_oov = 0
    dim_w = len(glove_embeddings['the'])
    vocab = {}
    for sent in train_set + test_set:
        tokens = sent2words(sent)
        for w in tokens:
            if w.lower() not in vocab:
                w_norm = w.lower()
                vocab[w_norm] = 1
                if w_norm in glove_embeddings:
                    word_emb = [float(ele) for ele in glove_embeddings[w_norm]]
                else:
                    word_emb = np.random.uniform(-0.25, 0.25, dim_w)
                    n_oov += 1
                # perform L2 normalization
                #embeddings[w_norm] = np.array(word_emb) / LA.norm(word_emb)
                embeddings[w_norm] = word_emb
    print "n_oov = %s" % n_oov
    if model_name == 'crf':
        if feat_name == 'embedding':
            crf_extractor(train_set=train_set, test_set=test_set, model_name=model_name, ds_name=ds_name, embeddings=embeddings)
        else:
            crf_extractor(train_set=train_set, test_set=test_set, model_name=model_name, ds_name=ds_name, embeddings=embeddings)
    elif model_name == 'svm':
        svm_extractor(train_set=train_set, test_set=test_set, embeddings=embeddings)
    elif model_name == 'lstm':
        lstm_extractor(train_set=train_set, test_set=test_set, embeddings=embeddings, win_size=3)
    elif model_name == 'conv_lstm':
        conv_lstm_extractor(train_set=train_set, test_set=test_set, embeddings=embeddings, win_size=3)
    elif model_name == 'ap':
        ap_extractor(train_set=train_set, test_set=test_set)


if __name__ == '__main__':
    dataset, model_name, feat_name = sys.argv[1:]
    run(ds_name=dataset, model_name=model_name, feat=feat_name)







