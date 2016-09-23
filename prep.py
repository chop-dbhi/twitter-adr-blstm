######################################
#
# prep.py
#
# Author: Anne Cocos
# Date:   May, 2016
#
# Prepare annotated ADR datasets for training/testing with
# our BLSTM RNN model.
#
# Creates a .pkl file that can be fed as input to adr_label.py
#
######################################

import cPickle as pkl
import os, sys
import csv
import re
import optparse
from utils import twokenize
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit

from settings import SETTINGS

Raw_data_dir = SETTINGS['SEQLAB_DATA_DIR']+'/raw'
Processed_data_dir = SETTINGS['SEQLAB_DATA_DIR']+'/processed'

def flatten(l):
    return [item for sublist in l for item in sublist]

def load_adefull(fname):
    if not os.path.isfile(fname):
        print 'Unable to find file', fname
        return None
    with open(fname, 'rb') as f:
        train_set, valid_set, test_set, dicts = pkl.load(f)
    return train_set, valid_set, test_set, dicts


def vectorize(listoftoklists, idxdict):
    '''
    Turn each list of tokens or labels in listoftoklists to an equivalent list of indices
    :param listoftoklists: list of lists
    :param idxdict: {tok->int}
    :return: list of np.array
    '''
    res = []
    for toklist in listoftoklists:
        res.append(np.array(map(lambda x: idxdict.get(x, idxdict['<UNK>']), toklist)).astype('int32'))
    return res


def read_clusters(cfile):
    '''
    Read clusters from file
    :param cfile: name of cluster file
    :return: {word -> cluster} dict
    '''
    return {l.split()[1]: l.split()[0] for l in open(cfile, 'rU').readlines()}


def parselist(strlist):
    '''
    Parse list from string representation of list
    :param strlist: string
    :return:list
    '''
    return [w[1:-1] for w in strlist[1:-1].split(', ')]

def create_adr_dataset(t, files, tokset, labelset):
    # Process data set
    tokset |= {'<UNK>'}
    raw_headers = ['id', 'start', 'end', 'semantic_type', 'span', 'reldrug', 'tgtdrug', 'text']
    atmention = re.compile('@\w+')

    try:
        os.makedirs(os.path.join(Processed_data_dir, t))
    except:
        pass

    for f in ['_'.join([d,t]) for d in files]:
        processed_rows = {}
        fout = open(os.path.join(Processed_data_dir, t, f), 'w')
        fnames = raw_headers+['tokens','labels','norm_text']
        wrt = csv.DictWriter(fout,fieldnames=fnames)
        wrt.writeheader()
        with open(os.path.join(Raw_data_dir, t, f), 'rU') as fin:
            dr = csv.DictReader(fin)
            for row in dr:
                # Pull from processed_rows dir so we can combine multiple annotations in a single tweet
                pr = processed_rows.get(row['id'], {h: row.get(h,[]) for h in fnames})

                # Tokenize
                tok_text = twokenize.tokenize(row['text'])
                tok_span = twokenize.tokenize(row['span'])
                # Add sequence labels to raw data
                labels = ['O'] * len(tok_text)
                if len(row['span']) > 0 and row['semantic_type'] != 'NEG':
                    s = row['semantic_type']
                    for i in range(len(tok_text)):
                        if tok_text[i:i+len(tok_span)] == tok_span:
                            labels[i] = labelset[s]['b']
                            if len(tok_span) > 1:
                                labels[i+1:i+len(tok_span)] = [labelset[s]['i']] * (len(tok_span)-1)

                # Combine spans and labels if duplicate
                pr['labels'] = combine_labels(pr['labels'], labels)
                if pr['span'] != row['span']:
                    pr['span'] = '|'.join([pr['span'], row['span']])
                pr['tokens'] = tok_text

                # Normalize text
                tok_text = [tt if not atmention.match(tt) else '<@MENTION>' for tt in tok_text]  # normalize @mentions
                lower_text = [w.lower() for w in tok_text]
                pr['norm_text'] = lower_text
                tokset |= set(lower_text)
                processed_rows[row['id']] = pr
        for pr, dct in processed_rows.items():
            wrt.writerow(dct)
        fout.close()
    return tokset

def comp_labels(l1,l2):
    if l1 != 'O':
        return l1
    elif l2 != 'O':
        return l2
    else:
        return 'O'

def combine_labels(seq_old, seq_new):
    if len(seq_old) == 0:
        return seq_new
    seq_combined = []
    for (o,n) in zip(seq_old, seq_new):
        seq_combined.append(comp_labels(o,n))
    return seq_combined

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-r", "--trainfiles", dest="trainfiles", default="asu_fullanno,chop_fullanno", help="Name(s) of data sets to use for training, comma separated")
    optparser.add_option("-t", "--testfiles", dest="testfiles",default="asu_fullanno,chop_fullanno", help="Name(s) of data sets to use for testing, comma separated")
    optparser.add_option("-o", "--outfile", dest="outfile", default="ade.full.pkl", help="Output pickle file")
    optparser.add_option("-v", "--validpct", dest="validpct", type=float, default=0.1, help="Fraction of training set to use as validation")
    (opts, _) = optparser.parse_args()

    if opts.validpct:
        assert opts.validpct

    if not os.path.isdir(Processed_data_dir):
        os.makedirs(Processed_data_dir)

    trainfiles = opts.trainfiles.split(',')
    testfiles = opts.testfiles.split(',')

    ''' Pre-process sentences in raw directory '''
    labelset = SETTINGS['labelset']

    tokset = {'<UNK>'}
    tokset |= create_adr_dataset('train', trainfiles, tokset, labelset)
    tokset |= create_adr_dataset('test', testfiles, tokset, labelset)

    # Build index dictionaries
    labels = ['O'] + sorted(list(set(flatten([subdict.values() for subdict in labelset.values()])))) + ['<UNK>']
    labels2idx = dict(zip(labels, range(1,len(labels)+1)))
    tok2idx = dict(zip(tokset, range(1,len(tokset)+1)))  # leave 0 for padding

    # Split train/validation from main train set
    train_toks_raw = []
    train_lex_raw = []
    train_y_raw = []
    valid_toks_raw = []
    valid_lex_raw = []
    valid_y_raw = []
    t_toks = []
    t_lex = []
    t_y = []
    t_class = []
    for dtype in trainfiles:
        with open(os.path.join(Processed_data_dir, 'train', dtype+'_train'), 'rU') as fin:
            rd = csv.DictReader(fin)
            for row in rd:
                t_toks.append(parselist(row['tokens']))
                t_lex.append(parselist(row['norm_text']))
                t_y.append(parselist(row['labels']))
                if '<UNK>' in parselist(row['labels']):
                    sys.stderr.write('<UNK> found in labels for tweet %s' % row['tokens'])
                t_class.append(row['semantic_type'])

    if opts.validpct > 0.0:
        sss = StratifiedShuffleSplit(t_class, 1, test_size=opts.validpct, random_state=0)
        for train_index, test_index in sss:
            train_toks_raw = [t_toks[i] for i in train_index]
            train_lex_raw = [t_lex[i] for i in train_index]
            train_y_raw = [t_y[i] for i in train_index]
            valid_toks_raw = [t_toks[i] for i in test_index]
            valid_lex_raw = [t_lex[i] for i in test_index]
            valid_y_raw = [t_y[i] for i in test_index]
    else:
        train_toks_raw = t_toks
        train_lex_raw = t_lex
        train_y_raw = t_y
        valid_toks_raw = []
        valid_lex_raw = []
        valid_y_raw = []

    test_toks_raw = []
    test_lex_raw = []
    test_y_raw = []
    for dtype in testfiles:
        with open(os.path.join(Processed_data_dir, 'test', dtype+'_test'), 'rU') as fin:
            rd = csv.DictReader(fin)
            for row in rd:
                test_toks_raw.append(parselist(row['tokens']))
                test_lex_raw.append(parselist(row['norm_text']))
                test_y_raw.append(parselist(row['labels']))
    # Convert each sentence of normalized tokens and labels into arrays of indices
    train_lex = vectorize(train_lex_raw, tok2idx)
    train_y = vectorize(train_y_raw, labels2idx)
    valid_lex = vectorize(valid_lex_raw, tok2idx)
    valid_y = vectorize(valid_y_raw, labels2idx)
    test_lex = vectorize(test_lex_raw, tok2idx)
    test_y = vectorize(test_y_raw, labels2idx)

    # Pickle the resulting data set
    with open(os.path.join(Processed_data_dir, opts.outfile),'w') as fout:
        pkl.dump([[train_toks_raw,train_lex,train_y],[valid_toks_raw,valid_lex,valid_y],[test_toks_raw,test_lex,test_y],
                  {'labels2idx':labels2idx, 'words2idx':tok2idx}], fout)
