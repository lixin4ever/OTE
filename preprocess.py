# coding: UTF-8
__author__ = 'lixin77'

from scrapy.selector import Selector
import cPickle
import nltk
from nltk import word_tokenize
from nltk.tag.stanford import StanfordPOSTagger
from nltk.parse.stanford import StanfordDependencyParser
import sys
import string
from utils import ot2bieos
import os


def build_pkl(ds, mode, schema="OT"):
    """
    schema: tagging schema in the sequence learning, OT refers to Targe and Outside of Target;
    BOIES refers to Begin, Outside, Inside, End, Singleton
    """
    path = './dataset/%s/%s.txt' % (mode, ds)
    print "process dataset: %s..." % ds
    ctx = 0
    records = []
    # Stanford POS Tagger
    pos_tagger = StanfordPOSTagger("english-bidirectional-distsim.tagger")
    # Stanford Dependency Parser
    dep_parser = StanfordDependencyParser(model_path=u'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')
    
    chunk_tags = []
    chunk_path = './dataset/%s/chunktags/%s_chunk.txt' % (mode, ds)
    with open(chunk_path) as fp:
        for line in fp:
            tags = line.strip().split()
            chunk_tags.append(tags)
    
    pos_tags = []
    pos_path = './dataset/%s/postags/%s_pos.txt' % (mode, ds)
    has_tagged = os.path.exists(pos_path)
    if has_tagged:
        with open(pos_path) as fp:
            tags = line.strip().split('\t')
            pos_tags.append(tags)

    dependencies = []
    dep_path = './dataset/%s/dependencies/%s_dep.txt' % (mode, ds)
    has_parsed = os.path.exists(dep_path)
    if has_parsed:
        with open(dep_path) as fp:
            text, deps = line.strip().split('####')
            triple_strings = deps.split('\t')
            triples = []
            for t in triple_strings:
                head, relation, tail = t.split('->')
                triples.append((head, relation, tail))
            dependencies.append(triples)
    pos_tag_lines = []
    dependency_lines = []
    i = 0
    with open(path) as fp:
        for line in fp:
            r = {}
            text, labelings = line.strip().split('####')
            word_seq = []
            tag_seq = []
            for ele in labelings.split(' '):
                items = ele.split('=')
                if len(items) > 2:
                    w = "%s:%s" % (items[0], items[1])
                    t = items[2]
                else:
                    w, t = ele.split('=')
                word_seq.append(w)
                tag_seq.append(t)
            if schema == 'BIEOS':
                tag_seq = ot2bieos(tag_sequence=tag_seq)
            # for debug
            #if i < 400:
            #    i += 1
            #    continue
            r['words'] = word_seq
            r['tags'] = tag_seq
            r['text'] = text

            if has_tagged:
                pos_res = pos_tags[i]
            else:
                pos_res = pos_tagger.tag(word_seq)
            if has_parsed:
                parse_triples = dependencies[i]
            else:
                if mode == 'sent':
                    parse_res = list(dep_parser.parse([unicode(w, encoding='utf-8') for w in word_seq]))[0]
                    # head->relation->tail
                    parse_triples = [(str(head[0]), str(relation), str(tail[0])) for (head, relation, tail)
                                     in list(parse_res.triples())]
                else:
                    # parse review sentence by sentence
                    parse_sent, parse_triples = [], []
                    for wid in xrange(len(word_seq)):
                        w = word_seq[wid]
                        if w == 'DELIM' or wid == len(word_seq) - 1:
                            parse_res = list(dep_parser.parse(parse_sent))[0]
                            parse_sent = []
                            parse_triples.extend([(str(head[0]), str(relation), str(tail[0])) for (head, relation, tail)
                                                  in list(parse_res.triples())])
                        else:
                            parse_sent.append(unicode(w, encoding='utf-8'))
            postag_seq = [tag for (w, tag) in pos_res]
            r['postags'] = postag_seq
            r['chunktags'] = chunk_tags[i]
            r['dependencies'] = parse_triples

            assert len(postag_seq) == len(word_seq) == len(tag_seq) == len(chunk_tags[i])
            records.append(r)

            if not has_tagged:
                pos_tag_lines.append('%s\n' % '\t'.join(postag_seq))
            if not has_parsed:
                triples = ['->'.join(list(t)) for t in parse_triples]
                dependency_lines.append('%s####%s\n' % (text, '\t'.join(triples)))

            ctx += 1
            if ctx % 100 == 0:
                print "process %s sentences" % ctx
            i += 1
    print "write back to the disk..."
    cPickle.dump(records, open('./pkl/%s/%s.pkl' % (mode, ds), 'w+'))

    if not has_tagged:
        with open(pos_path, 'w+') as fp:
            fp.writelines(pos_tag_lines)

    if not has_parsed:
        with open(dep_path, 'w+') as fp:
            fp.writelines(dependency_lines)


def process_text(text):
    """
    process the text and filter some special symbol
    :param text:
    :return:
    """
    # string preprocessing and aspect term will not be processed
    dot_exist = ('.' in text)
    cur_text = text.replace('.', '')
    #cur_text = cur_text.replace('-', ' ')
    cur_text = cur_text.replace(' - ', ', ').strip()
    cur_text = cur_text.replace('- ', ' ').strip()

    # split words and punctuations
    if '? ' not in cur_text:
        cur_text = cur_text.replace('?', '? ').strip()
    if '! ' not in cur_text:
        cur_text = cur_text.replace('!', '! ').strip()
    cur_text = cur_text.replace('(', '')
    cur_text = cur_text.replace(')', '')
    cur_text = cur_text.replace('...', ', ').strip('.').strip().strip(',')
    # remove quote
    cur_text = cur_text.replace('"', '')
    cur_text = cur_text.replace(" '", " ")
    cur_text = cur_text.replace("' ", " ")

    cur_text = cur_text.replace(':', ', ')
    if dot_exist:
        cur_text += '.'
    # correct some typos
    cur_text = cur_text.replace('cant', "can't")
    cur_text = cur_text.replace('wouldnt', "wouldn't")
    cur_text = cur_text.replace('dont', "don't")
    cur_text = cur_text.replace('didnt', "didn't")
    cur_text = cur_text.replace("you 're", "you're")

    # replace some special symbol
    cur_text = cur_text.replace(u' – ', ', ').strip()

    cur_text = cur_text.replace(u"‘", "")
    # filter the non-ascii character
    cur_text = ''.join([ch if ord(ch) < 128 else ' ' for ch in cur_text])
    return cur_text

def extract_aspect(aspects, text, dataset_name):
    """
    extract aspects from xml tags
    :param aspects: a list of aspect tags / selectors
    :param text: corresponding sentence
    :param dataset_name: name of dataset
    :return:
    """
    counter = 0
    id2aspect = {}
    # number of aspects, singleton, multi-word-aspects in the sentence, respectively
    n_aspect, n_singleton, n_mult_word = 0, 0, 0
    cur_text = text
    from_to_pairs = []
    for t in aspects:
        _from = int(t.xpath('.//@from').extract()[0])
        _to = int(t.xpath('.//@to').extract()[0])
        if _from == 0 and _to == 0:
            # NULL target
            continue
        if not dataset_name.startswith('14'):
            target = t.xpath('.//@target').extract()[0].replace(u'\xa0', ' ')
        else:
            target = t.xpath('.//@term').extract()[0].replace(u'\xa0', ' ')
        if target == 'NULL':
            # there is no aspect in the text
            continue
        flag = False
        # remove special symbol in aspect term
        target = target.replace(u'é', 'e')
        target = target.replace(u'’', "'")
        if text[_from:_to] == target:
            flag = True
        elif (_from - 1 >= 0) and text[(_from - 1):(_to - 1)] == target:
            _from -= 1
            _to -= 1
            flag = True
        elif (_to + 1 < len(text)) and text[(_from + 1):(_to + 1)] == target:
            _from += 1
            _to += 1
            flag = True
        # we can find the aspect in the raw text
        assert flag

        if (_from, _to) in from_to_pairs:
            continue
        aspect_temp_value = 'ASPECT%s' % counter
        counter += 1
        id2aspect[aspect_temp_value] = target
        cur_text = cur_text.replace(target, aspect_temp_value)
        from_to_pairs.append((_from, _to))
        n_aspect += 1
        if len(target.split()) > 1:
            n_mult_word += 1
        else:
            n_singleton += 1
    return id2aspect, n_aspect, n_singleton, n_mult_word, cur_text

def format_output(x, y, text):
    """
    format the dataset output
    :param x: word sequence
    :param y: tag sequence
    :param text: raw text
    :return:
    """
    tag_sequence = ''
    for i in xrange(len(x)):
        if i == (len(x) - 1):
            tag_sequence = '%s%s=%s' % (tag_sequence, x[i], y[i])
        else:
            tag_sequence = '%s%s=%s ' % (tag_sequence, x[i], y[i])
    data_line = '%s####%s\n' % (text, tag_sequence)
    return data_line.encode('utf-8')

def extract_text(dataset_name, mode='sent'):
    """
    extract textual information from the xml file
    :param dataset_name: dataset name
    :param mode: extract mode, 'sent' denotes extract sent, 'review' denotes extract whole review / document
    """
    delset = string.punctuation
    fpath = './raw_data/%s.xml' % dataset_name

    page_source = ''
    with open(fpath) as fp:
        for line in fp:
            page_source = '%s%s' % (page_source, line.strip())
    reviews = []
    if mode == 'sent':
        sentences = Selector(text=page_source).xpath('//sentences/sentence')
        reviews = [sentences]
    elif mode == 'doc':
        review_eles = Selector(text=page_source).xpath('//review')
        print "n_reviews", len(review_eles)
        for ele in review_eles:
            sentences = ele.xpath('.//sentence')
            # single review may contain multiple sent
            reviews.append(sentences)
    n_sen = 0
    n_word = 0
    # number of aspects, singletons and multi-words in the dataset, respectively
    n_aspect, n_singleton, n_mult_word = 0, 0, 0
    n_sen_with_no_aspect = 0
    lines = []
    for sentences in reviews:
        # scan all of the reviews
        x, y, review_text = [], [], ''
        for sid in xrange(len(sentences)):
            sen = sentences[sid]
            prev = ''
            n_sen += 1
            text = sen.xpath('.//text/text()').extract()[0]
            text = text.replace(u'\xa0', ' ')
            # note: preprocessing in the raw text should not change the index
            text = text.replace(u'é', 'e')
            text = text.replace(u'’', "'")
            cur_text = text

            assert isinstance(dataset_name, str)
            if dataset_name.startswith('14'):
                aspects = sen.xpath('.//aspectterms/aspectterm')
            else:
                aspects = sen.xpath('.//opinions/opinion')

            if not aspects:
                # sent with no aspect
                n_sen_with_no_aspect += 1
            else:
                id2aspect, n_a, n_s, n_m, cur_text = extract_aspect(aspects=aspects, text=cur_text, dataset_name=dataset_name)
                n_aspect += n_a
                n_singleton += n_s
                n_mult_word += n_m
            if mode == 'sent':
                # flush output buffer every sentence
                x, y = [], []
            # process the text and filter the unnecessary characters
            cur_text = process_text(text=cur_text)
            tokens = word_tokenize(cur_text)
            for t in tokens:
                if t.startswith('ASPECT'):
                    # in this case, t is actually the identifier of aspect
                    raw_string = id2aspect[t[:7]]
                    aspect_words = raw_string.split()
                    n_aw = len(aspect_words)
                    x.extend(aspect_words)
                    y.extend(['T'] * n_aw)
                    n_word += n_aw
                else:
                    # t is the literal value
                    if not t.strip() == '':
                        # t is not blank space or empty string
                        x.append(t.strip())
                        y.append('O')
                        n_word += 1
            # length check for every sentence
            assert len(x) == len(y)
            if mode == 'sent':
                # write back after processing a sentence
                lines.append(format_output(x=x, y=y, text=text))
            else:
                # mode == 'doc'
                if review_text != '':
                    review_text = '%s %s' % (review_text, text)
                else:
                    review_text = text
                if sid == len(sentences) - 1:
                    # write after processing a review
                    lines.append(format_output(x=x, y=y, text=review_text))
                else:
                    # add sentence delimiter after the internal sentence
                    x.append('DELIM')
                    y.append('O')

    with open('./dataset/%s/%s.txt' % (mode, dataset_name), 'w+') as fp:
        fp.writelines(lines)

    print "dataset:", dataset_name
    print "n_sen:", n_sen
    print "average length:", int(n_word / n_sen)
    print "total aspects:", n_aspect
    print "n_singleton:", n_singleton
    print "n_mult_words:", n_mult_word
    print "n_without_aspect:", n_sen_with_no_aspect
    print "n_tokens:", n_word
    print "\n\n"


if __name__ == '__main__':
    #ds = sys.argv[1]
    #ds = '14semeval_laptop_train'
    ds, mode = 'all', 'doc'
    if ds == 'all':
        for ds in ['15semeval_rest_test', '15semeval_rest_train',
        '14semeval_laptop_train', '14semeval_laptop_test',
        '14semeval_rest_train', '14semeval_rest_test',
        '16semeval_rest_train', '16semeval_rest_test']:
            if ds.startswith('14') and mode == 'doc':
                continue
            #extract_text(dataset_name=ds, mode=mode)
            build_pkl(ds=ds, mode=mode)
    else:
        #extract_text(dataset_name=ds, mode='sent')
        build_pkl(ds=ds, mode=mode)


