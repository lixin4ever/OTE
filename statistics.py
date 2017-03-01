import os, sys
from utils import *
import cPickle


def find_non_subj_sent(ds_name):
    train_set = cPickle.load(open('./pkl/%s_train.pkl' % ds_name, 'rb'))

    test_set = cPickle.load(open('./pkl/%s_test.pkl' % ds_name, 'rb'))

    n_non_subj_train, n_subj_train = 0, 0
    n_non_subj_test, n_subj_test = 0, 0

    display_count = 0
    for sent in train_set:
        tags = sent2tags(sent)
        if 'T' not in tags:
            n_non_subj_train += 1
        else:
            n_subj_train += 1
    for sent in test_set:
        tags = sent2tags(sent)
        if 'T' not in tags:
            n_non_subj_test += 1
            words = sent2words(sent)
            if display_count < 10:
                print ' '.join(words)
                display_count += 1
        else:
            n_subj_test += 1

    return n_subj_train, n_subj_test, n_non_subj_train, n_non_subj_test


if __name__ == '__main__':
    ds_name = sys.argv[1]
    n_subj_train, n_subj_test, n_non_subj_train, n_non_subj_test = find_non_subj_sent(ds_name)
    print "dataset %s:" % ds_name
    print "training set: subj=%s, nsubj=%s" % (n_subj_train, n_non_subj_train)
    print "testing set: subj=%s, nsubj=%s" % (n_subj_test, n_non_subj_test)

