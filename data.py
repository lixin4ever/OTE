__author__ = 'lixin77'


class Token(object):
    """
    single word
    """
    def __init__(self, surface):
        # observation features, e.g., word identity, prefix of word, suffix of word, pos tag...
        self.observations = []
        # surface name of the Token
        self.surface = surface

    def add(self, features):
        """
        add observation features
        """
        self.observations.extend(features)

class Sequence(object):
    """
    sequence
    """
    def __init__(self, tokens, truth):
        """

        :param tokens: input sequence
        :param truth: tags of input sequence
        :return:
        """
        self.tokens = list(tokens)
        self.tags = truth
        # sequence length
        self.N = len(tokens)
        self.feature_table = None
        self.target_features = None

    def F(self, t, yt_1, yt):
        """

        :param t: current time step
        :param yt_1: y[t-1]
        :param yt: y[t]
        :return:
        """
        # label-token / label-observation / state-observation features, also called emission features in HMM
        for o in self.tokens[t].observations:
            yield '%s-%s' % (yt, o)
        # label / state features
        yield '%s' % yt
        # label-label / state-state features, also called transition features in HMM
        yield '%s-%s' % (yt_1, yt)

class Feature(object):
    """
    bijective mapping feature and feature index
    """
    def __init__(self, random_int=None):
        """
        """
        # mapping between feature and feature id
        self._mapping = {}
        # mapping between feature id and feature
        self._indexedMapping = {}
        self.idx = 0
        self._frozen = False
        self._growing = True
        self._random_int = random_int

    def keys(self):
        return self._mapping.iterkeys()

    def items(self):
        return self._mapping.iteritems()

    def imap(self, seq):
        """
        """
        for s in seq:
            # index term is a feature string
            x = self[s]
            if x is not None:
                yield x

    def map(self, seq):
        return list(self.imap(seq))



    def add_many(self, x):
        for k in x:
            self.add(x)

    def __getitem__(self, k):
        """
        """
        try:
            return self._mapping[k]
        except KeyError:
            # met a new key, create an item in the map manually
            if self._frozen:
                raise ValueError('froze! Element addition is invalid!')
            if not self._growing:
                return None
            x = self._mapping[k] = self.idx
            self._indexedMapping[x] = k
            self.idx += 1
            return x
    add = __getitem__

    def __setitem__(self, key, value):
        assert key not in self._mapping
        assert isinstance(value, int)
        self._mapping[key] = value
        self._indexedMapping[value] = key

    def __len__(self):
        return len(self._mapping)

    def __iter__(self):
        for i in xrange(len(self)):
            yield self._indexedMapping[i]

    def enum(self):
        for i in xrange(len(self)):
            yield (i, self._indexedMapping[i])

    def __eq__(self, other):
        return self._mapping == other._mapping

    def __contains__(self, k):
        return k in self._mapping

    def lookup(self, i):
        if i is None:
            return None
        return self._indexedMapping[i]

    def lookup_many(self, x):
        return map(self.lookup, x)


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


def token_features(w):
    """
    generate observation features, follow the idea in https://github.com/ppfliu/opinion-target/blob/master/absa.py
    """
    # word identity
    yield 'word=%s' % w
    w_shape = get_shape(w)
    yield 'word_shape=%s' % w_shape
    # lower- or upper-case
    yield 'word.lower()=%s' % w.lower()
    yield 'word.upper()=%s' % w.upper()

    # is-xxx feature
    yield 'word.isdigit()=%s' % w.isdigit()
    yield 'word.isupper()=%s' % w.isupper()
    yield 'word.isalpha()=%s' % w.isalpha()
    yield 'word.istitle()=%s' % w.istitle()

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
    yield 'prefix3=%s' % p3
    yield 'suffix3=%s' % s3
    yield 'prefix2=%s' % p2
    yield 'suffix2=%s' % s2
    yield 'prefix1=%s' % p1
    yield 'suffix1=%s' % s1