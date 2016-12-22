__author__ = 'lixin77'
import cPickle
import sklearn_crfsuite
from sklearn_crfsuite import metrics, scorers
from sklearn.svm import SVC
import sys
from utils import *
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Embedding, TimeDistributed, Dense, Flatten
from keras.callbacks import EarlyStopping

#embeddings = {}
def crf_extractor(train_set, test_set, embeddings=None):
    """
    linear-chain crf extractor with basic feature template
    :param train_set: training dataset
    :param test_set: testing dataset
    """
    train_Y = [sent2tags(sent) for sent in train_set]
    test_Y = [sent2tags(sent) for sent in test_set]

    if not embeddings:
        print "crf with word-level features..."
        train_X = [sent2features(sent) for sent in train_set]
        test_X = [sent2features(sent) for sent in test_set]


    else:
        print "crf with word embeddings..."
        train_words = [sent2tokens(sent) for sent in train_set]
        #train_Y = [sent2tags(sent) for sent in train_set]
        train_words = to_lower(word_seqs=train_words)

        test_words = [sent2tokens(sent) for sent in test_set]
        test_words = to_lower(word_seqs=test_words)
        #test_Y = [sent2tags(sent) for sent in test_set]

        vocab, df, max_len = get_corpus_info(trainset=train_words, testset=test_words)

        train_words_norm = [normalize(seq, df) for seq in train_words]
        test_words_norm = [normalize(seq, df) for seq in test_words]

        dim_w = len(embeddings['the'])
        embeddings['DIGIT'] = np.random.uniform(-0.25, 0.25, dim_w)
        embeddings['UNKNOWN'] = np.random.uniform(-0.25, 0.25, dim_w)

        train_X = [sent2embeddings(sent, embeddings) for sent in train_words]
        test_X = [sent2embeddings(sent, embeddings) for sent in test_words]


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

    print evaluate_chunk(test_Y=test_Y, pred_Y=pred_Y)

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

def lstm_extractor(train_set, test_set, embeddings):
    """
    LSTM extractor for aspect term extraction, text pre-processing step follows the
    paper: http://www.aclweb.org/anthology/D/D15/D15-1168.pdf
    :param train_set:
    :param test_set:
    :param embeddings:
    """
    print "Bi-directional LSTM with word embeddings..."
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
    n_w = len(vocab)
    vocab['PADDING'] = 0
    if not 'DIGIT' in vocab:
        vocab['DIGIT'] = n_w + 1
        n_w += 1
    if not 'UNKNOWN' in vocab:
        vocab['UNKNOWN'] = n_w + 1
        n_w += 1

    embeddings_weights = np.zeros((n_w + 1, dim_w))
    for (w, idx) in vocab.items():
        embeddings_weights[idx, :] = embeddings[w]
    train_X, train_Y = token2identifier(X=train_words_norm, Y=train_tags, vocab=vocab)
    train_X, train_Y = padding_seq(train_X, train_Y, max_len=max_len)
    print "train shape:", train_X.shape

    test_X, test_Y = token2identifier(X=test_words_norm, Y=test_tags, vocab=vocab)
    test_X, test_Y = padding_seq(test_X, test_Y, max_len=max_len)
    print "test shape:", test_X.shape

    print "Build the Bi-LSTM model..."
    LSTM_extractor = Sequential()
    LSTM_extractor.add(Embedding(output_dim=dim_w, input_dim=n_w + 1, weights=[embeddings_weights]))
    LSTM_extractor.add(Bidirectional(LSTM(100, return_sequences=True), merge_mode='concat', input_shape=(max_len, 300)))
    #LSTM_extractor.add(LSTM(100, return_sequences=True))
    LSTM_extractor.add(TimeDistributed(Dense(output_dim=1, activation='sigmoid')))
    #LSTM_extractor.add(Flatten())
    LSTM_extractor.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #print LSTM_extractor.summary()
    print "Begin to training the model..."
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    LSTM_extractor.fit(train_X, train_Y, batch_size=32, nb_epoch=30,
                       validation_data=(test_X, test_Y), callbacks=[early_stopping])

    res = LSTM_extractor.predict_classes(test_X)
    res = res.reshape((res.shape[0], res.shape[1] * res.shape[2]))

    print "output dim:", res.shape
    assert res.shape == test_X.shape
    assert res.shape[0] == len(test_X)
    pred_tags = []
    for (i, raw_seq) in enumerate(test_tags):
        pred_tags.append(get_valid_seq(padded_seq=res[i], raw_len=len(raw_seq)))

    print evaluate_chunk(test_Y=test_tags, pred_Y=pred_tags)

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
    print "load word embeddings..."
    with open('./embeddings/glove_6B_300d.txt', 'r') as fp:
        for line in fp:
            values = line.strip().split()
            word, vec = values[0], values[1:]
            glove_embeddings[word] = vec

    dim_w = len(glove_embeddings['the'])
    vocab = {}
    for sent in train_set + test_set:
        tokens = sent2tokens(sent)
        for w in tokens:
            if w.lower() not in vocab:
                w_norm = w.lower()
                vocab[w_norm] = 1
                if w_norm in glove_embeddings:
                    word_emb = [float(ele) for ele in glove_embeddings[w_norm]]
                else:
                    word_emb = np.random.uniform(-0.25, 0.25, dim_w)
                embeddings[w_norm] = word_emb
    if model_name == 'crf':
        if feat_name == 'embedding':
            crf_extractor(train_set=train_set, test_set=test_set, embeddings=embeddings)
        else:
            crf_extractor(train_set=train_set, test_set=test_set)
    elif model_name == 'svm':
        svm_extractor(train_set=train_set, test_set=test_set, embeddings=embeddings)
    elif model_name == 'lstm':
        lstm_extractor(train_set=train_set, test_set=test_set, embeddings=embeddings)


if __name__ == '__main__':
    dataset, model_name, feat_name = sys.argv[1:]
    run(ds_name=dataset, model_name=model_name, feat=feat_name)







