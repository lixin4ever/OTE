# coding: UTF-8
__author__ = 'lixin77'

from scrapy.selector import Selector
import cPickle
import nltk
from nltk import word_tokenize
from nltk.tag.stanford import StanfordPOSTagger
import sys
import string
from utils import ot2bieos


def build_pkl(ds, schema="OT"):
    """
    schema: tagging schema in the sequence learning, OT refers to Targe and Outside of Target;
    BOIES refers to Begin, Outside, Inside, End, Singleton
    """
    path = './dataset/%s.txt' % ds
    print "process dataset: %s..." % ds
    ctx = 0
    records = []
    tagger = StanfordPOSTagger("english-bidirectional-distsim.tagger")
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
            r['words'] = word_seq
            r['tags'] = tag_seq
            r['text'] = text
            res = tagger.tag(word_seq)
            postag_seq = [tag for (w, tag) in res]
            r['postags'] = postag_seq
            assert len(postag_seq) == len(word_seq) == len(tag_seq)
            records.append(r)
            ctx += 1
            if ctx % 100 == 0:
                print "process %s sentences" % ctx
    print "write back to the disk..."
    cPickle.dump(records, open('./pkl/%s.pkl' % ds, 'w+'))


def extract_text(dataset_name):
    """
    extract textual information from the xml file
    dataset_name: name of dataset
    """
    delset = string.punctuation
    fpath = './raw_data/%s.xml' % dataset_name

    page_source = ''
    with open(fpath) as fp:
        for line in fp:
            page_source = '%s%s' % (page_source, line.strip())
    sentences = Selector(text=page_source).xpath('//sentences/sentence')
    n_sen = 0
    n_word = 0
    n_aspect = 0
    n_sen_with_no_aspect = 0
    n_singleton = 0
    n_mult_word = 0
    lines = []
    for sen in sentences:
        prev = ''
        n_sen += 1
        text = sen.xpath('.//text/text()').extract()[0]
        text = text.replace(u'\xa0', ' ')
        cur_text = text

        assert isinstance(dataset_name, str)
        if dataset_name.startswith('14'):
            aspects = sen.xpath('.//aspectterms/aspectterm')
        else:
            aspects = sen.xpath('.//opinions/opinion')
        from_to_pairs = []
        if not aspects:
            # sentences with no aspect
            n_sen_with_no_aspect += 1
        else:
            counter = 0
            id2aspect = {}
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
        y = []
        x = []
        # string preprocessing and aspect term will not be processed
        dot_exist = ('.' in cur_text)
        cur_text = cur_text.replace('.', '')
        #cur_text = cur_text.replace('-', ' ')
        cur_text = cur_text.replace(' - ', ', ').strip()
        #cur_text = cur_text.replace(' – ', ', ').strip()
        cur_text = cur_text.replace(u' – ', ', ').strip()
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
        cur_text = cur_text.replace(':', ', ')
        if dot_exist:
            cur_text += '.'
        # correct some typos
        cur_text = cur_text.replace('cant', "can't")
        cur_text = cur_text.replace('wouldnt', "wouldn't")
        cur_text = cur_text.replace('dont', "don't")
        cur_text = cur_text.replace('didnt', "didn't")

        #tokens = cur_text.split()
        tokens = word_tokenize(cur_text)

        for t in tokens:
            if t.startswith('ASPECT'):
                # in this case, t is actually the identifier of aspect
                raw_string = id2aspect[t[:7]]
                aspect_words = raw_string.split()
                for aw in aspect_words:
                    x.append(aw)
                    y.append('T')
                    n_word += 1
            else:
                # t is the literal value
                if not t.strip() == '':
                    # t is not blank space or empty string
                    x.append(t.strip())
                    y.append('O')
                    n_word += 1
        assert len(x) == len(y)
        tag_sequence = ''
        for i in xrange(len(x)):
            if i == (len(x) - 1):
                tag_sequence = '%s%s=%s' % (tag_sequence, x[i], y[i])
            else:
                tag_sequence = '%s%s=%s ' % (tag_sequence, x[i], y[i])
        data_line = '%s####%s\n' % (text, tag_sequence)
        data_line = data_line.encode('utf-8')
        lines.append(data_line)
    with open('./dataset/%s.txt' % dataset_name, 'w+') as fp:
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
    ds = 'all'
    if ds == 'all':
        for ds in ['14semeval_laptop_train', '14semeval_laptop_test', '14semeval_rest_train', '14semeval_rest_test',
                   '15semeval_rest_train', '15semeval_rest_test', '16semeval_rest_train', '16semeval_rest_test']:
            extract_text(dataset_name=ds)
            #build_pkl(ds=ds)
    else:
        extract_text(dataset_name=ds)
        #build_pkl(ds=ds)


