__author__ = 'lixin77'


class Token(object):
    """
    single word
    """
    def __init__(self, surface, pos=None, chunk=None, dep=None, embedding=None):
        """

        :param surface: surface name (i.e., identity) of token
        :param pos: part-of-speech tags
        :param chunk: chunk tags (phrase-level features)
        :param dep: dependency parsing features
        :param embedding: embedding features
        """
        # observation features, e.g., word identity, prefix of word, suffix of word, pos tag...
        self.observations = []
        # surface name of the Token
        self.surface = surface
        self.pos = pos
        self.chunk = chunk
        self.dep = dep
        self.embedding = embedding

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
        feature function for obtaining the features at each timestep t
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

class FeatureIndexer(object):
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
            self.add(k)

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





