__author__ = 'lixin77'
import cPickle
import sklearn_crfsuite
from sklearn_crfsuite import metrics, scorers
from sklearn.svm import SVC
import sys
from utils import *
import numpy as np

#embeddings = {}
def crf_extractor(train_set, test_set):
    """
    linear-chain crf extractor with basic feature template
    :param train_set: training dataset
    :param test_set: testing dataset
    """
    print "feature transformation..."
    train_X = [sent2features(sent) for sent in train_set]
    test_X = [sent2features(sent) for sent in test_set]

    train_Y = [sent2tags(sent) for sent in train_set]
    test_Y = [sent2tags(sent) for sent in test_set]

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

def svm_extractor(train_set, test_set, embeddings):
    """
    window-based support vector machine for aspect extraction
    :param train_set: training set
    :param test_set: testing set
    :param embeddings: pretrained word_embeddings
    """
    train_words = [sent2tokens(sent) for sent in train_set]
    train_tags = [sent2tags(sent) for sent in train_set]
    train_words = to_lower(word_seqs=train_words)

    test_words = [sent2tokens(sent) for sent in test_set]
    test_words = to_lower(word_seqs=test_words)
    test_tags = [sent2tags(sent) for sent in test_set]

    vocab, df = get_corpus_info(trainset=train_set, testset=test_set)

    train_words_norm = [normalize(seq, df) for seq in train_words]
    test_words_norm = [normalize(seq, df) for seq in test_words]

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
    pass



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
    with open('glove_6B_300d.txt', 'r') as fp:
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
        crf_extractor(train_set=train_set, test_set=test_set)
    elif model_name == 'svm':
        svm_extractor(train_set=train_set, test_set=test_set, embeddings=embeddings)


if __name__ == '__main__':
    dataset, model_name = sys.argv[1:]
    run(ds_name=dataset, model_name=model_name, feat='word')







