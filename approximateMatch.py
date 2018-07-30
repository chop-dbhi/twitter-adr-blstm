from keras.callbacks import Callback
import os
import numpy as np

class ApproximateMatch(Callback):
    ''' Calculate Approximate Matching statistics

    This callback generates Approximate Matching stats using the model's prediction
    on the validation data after each epoch.

    # Arguments
        valid_toks: actual tokens (words) in validation set; list of lists length N
        valid_y: actual classes in validation set; list of numpy arrays length N
        X_valid: vectorized validation samples, dimensions (N*padlen x width x vocabsize)
        padlen: padded length of each sentence
        idx2label: dictionary with label indices as keys and label names as values
        pred_dir: path to the directory where this should write the predictions after
            each epoch, which will be scored
    '''
    def __init__(self, valid_toks, valid_y, X_valid, padlen, idx2label, pred_dir='./predictions'):
        super(Callback, self).__init__()
        self.valid_toks = valid_toks
        self.valid_y = valid_y
        self.X_valid = X_valid
        self.padlen = padlen
        assert X_valid.shape[0] == padlen * len(valid_toks)
        self.window = X_valid.shape[1]

        self.idx2label = idx2label
        self.pred_dir = pred_dir
        try:
            os.makedirs(pred_dir)
        except:
            pass


    def on_epoch_end(self, epoch, logs={}):

        pred = self.model.predict_classes(self.X_valid)


        N = len(self.valid_toks)
        fname = os.path.join(self.pred_dir, 'approxmatch_epoch'+str(epoch))
        with open(fname, 'w') as fout:
            for i in range(N):
                fout.write('BOS\tO\tO\n')

                sentlen = len(self.valid_toks[i])
                preds = [self.idx2label[j] for j in pred[i*self.padlen:i*self.padlen+sentlen]]
                actuals = [self.idx2label[j] for j in self.valid_y[i]]
                for (w, act, p) in zip(self.valid_toks[i], actuals, preds):
                    fout.write('\t'.join([w, act, p])+'\n')

                fout.write('EOS\tO\tO\n')
        scores = self.get_approx_match(fname)

        with open(fname, 'a') as fout:
            fout.write('\nApproximate Matching Results:\n  ADR: Precision '+ str(scores['p'])+ ' Recall ' + str(scores['r']) + ' F1 ' + str(scores['f1']))

    def get_approx_match(self, filename):
        ''' Calculate approximate match from results file, written with format:
        word <actualLabel> <predictedLabel>

        Sentences should be demarcated by 'BOS' and 'EOS' lines.
        Labels can be one of 'O', 'B-ADR', 'I-ADR', 'B-Indication', 'I-Indication'

        :param filename: Name of file with results
        :return:
        '''
        return get_approx_match(filename)

class ApproximateMatch_SEQ(Callback):
    ''' Calculate Approximate Matching statistics

    This callback generates Approximate Matching stats using the model's prediction
    on the validation data after each epoch.

    # Arguments
        valid_toks: actual tokens (words) in validation set; list of lists length N
        valid_y: actual classes in validation set; list of numpy arrays length N
        X_valid: vectorized validation samples, dimensions (N x padlen x vocabsize)
        padlen: padded length of each sentence/sequence
        idx2label: dictionary with label indices as keys and label names as values
        pred_dir: path to the directory where this should write the predictions after
            each epoch, which will be scored
    '''
    def __init__(self, valid_toks, valid_y, valid_x, idx2label, pred_dir='./predictions', fileprefix=''):
        super(Callback, self).__init__()
        self.valid_toks = valid_toks
        self.valid_y = valid_y
        self.valid_x = valid_x
        self.padlen = valid_x.shape[1]
        self.fileprefix=fileprefix

        self.idx2label = idx2label
        self.pred_dir = pred_dir
        try:
            os.makedirs(pred_dir)
        except:
            pass


    def on_epoch_end(self, epoch, logs={}):

        ## BIDIRECTIONAL
        pred_probs = self.model.predict(self.valid_x)
        pred = np.argmax(pred_probs, axis=2)

        N = len(self.valid_toks)
        fname = os.path.join(self.pred_dir, self.fileprefix+'approxmatch_epoch'+str(epoch))
        with open(fname, 'w') as fout:
            for i in range(N):
                fout.write('BOS\tO\tO\n')

                sentlen = len(self.valid_toks[i])
                startind = self.padlen - sentlen

                preds = [self.idx2label[j] for j in pred[i][startind:]]
                actuals = [self.idx2label[j] for j in np.argmax(self.valid_y[i], axis=1)[startind:]]
                for (w, act, p) in zip(self.valid_toks[i], actuals, preds):
                    fout.write('\t'.join([w, act, p])+'\n')

                fout.write('EOS\tO\tO\n')
        scores = self.get_approx_match(fname)

        with open(fname, 'a') as fout:
            fout.write('\nApproximate Matching Results:\n  ADR: Precision '+ str(scores['p'])+ ' Recall ' + str(scores['r']) + ' F1 ' + str(scores['f1']))

    def get_approx_match(self, filename):
        ''' Calculate approximate match from results file, written with format:
        word <actualLabel> <predictedLabel>

        Sentences should be demarcated by 'BOS' and 'EOS' lines.
        Labels can be one of 'O', 'B-ADR', 'I-ADR', 'B-Indication', 'I-Indication'

        :param filename: Name of file with results
        :return:
        '''
        return get_approx_match(filename)


def find_inds(lst, item):
    return [i for i, x in enumerate(lst) if x == item]

def approx_match(l1,l2):
    return len([l for l in l1 if l in l2]) > 0

def get_approx_match(filename):
    ''' Calculate approximate match from results file, written with format:
    word <actualLabel> <predictedLabel>

    Sentences should be demarcated by 'BOS' and 'EOS' lines.
    Labels can be one of 'O', 'B-ADR', 'I-ADR', 'B-Indication', 'I-Indication'

    :param filename: Name of file with results
    :return:
    '''
    def find_inds(lst, item):
        return [i for i, x in enumerate(lst) if x == item]

    def approx_match(l1,l2):
        return len([l for l in l1 if l in l2]) > 0

    auto_seqs = []
    hand_seqs = []

    # read in sequences
    with open(filename, 'rU') as fin:
        a_s = []
        h_s = []
        for line in fin:
            if len(line.strip()) == 0:
                continue
            word, h, a = line.strip().split()
            if word == 'BOS':
                a_s = ['O']
                h_s = ['O']
            elif word == 'EOS':
                auto_seqs.append(a_s)
                hand_seqs.append(h_s)
            else:
                a_s.append(a)
                h_s.append(h)
    # count tags and matches
    auto_tags_adr = 0
    auto_tags_indic = 0

    hand_tags_adr = 0
    hand_tags_indic = 0

    matches_adr = 0
    matches_indic = 0
    for a_seq,h_seq in zip(auto_seqs, hand_seqs):
        a_cntr_adr = [0] * len(a_seq)
        a_cntr_indic = [0] * len(a_seq)
        h_cntr_adr = [0] * len(h_seq)
        h_cntr_indic = [0] * len(h_seq)
        a_mask_adr = [0] * len(a_seq)
        a_mask_indic = [0] * len(a_seq)
        h_mask_adr = [0] * len(h_seq)
        h_mask_indic = [0] * len(h_seq)
        for i in range(1, len(a_seq)):
            if (a_seq[i-1] == 'O' and 'ADR' in a_seq[i]) or a_seq[i] == 'B-ADR':
                a_cntr_adr[i] = a_cntr_adr[i-1] + 1
            else:
                a_cntr_adr[i] = a_cntr_adr[i-1]
            if (a_seq[i-1] == 'O' and 'Indication' in a_seq[i]) or a_seq[i] == 'B-Indication':
                a_cntr_indic[i] = a_cntr_indic[i-1] + 1
            else:
                a_cntr_indic[i] = a_cntr_indic[i-1]

            if (h_seq[i-1] == 'O' and 'ADR' in h_seq[i]) or h_seq[i] == 'B-ADR':
                h_cntr_adr[i] = h_cntr_adr[i-1] + 1
            else:
                h_cntr_adr[i] = h_cntr_adr[i-1]
            if (h_seq[i-1] == 'O' and 'Indication' in h_seq[i]) or h_seq[i] == 'B-Indication':
                h_cntr_indic[i] = h_cntr_indic[i-1] + 1
            else:
                h_cntr_indic[i] = h_cntr_indic[i-1]

            a_mask_adr[i] = 'ADR' in a_seq[i]
            a_mask_indic[i] = 'Indication' in a_seq[i]
            h_mask_adr[i] = 'ADR' in h_seq[i]
            h_mask_indic[i] = 'Indication' in h_seq[i]
        a_cntr_adr = [a*m for a,m in zip(a_cntr_adr,a_mask_adr)]
        a_cntr_indic = [a*m for a,m in zip(a_cntr_indic,a_mask_indic)]

        h_cntr_adr = [h*m for h,m in zip(h_cntr_adr,h_mask_adr)]
        h_cntr_indic = [h*m for h,m in zip(h_cntr_indic,h_mask_indic)]

        auto_tags_adr += max(a_cntr_adr)
        auto_tags_indic += max(a_cntr_indic)
        hand_tags_adr += max(h_cntr_adr)
        hand_tags_indic += max(h_cntr_indic)
        a_subseqs_adr = [find_inds(a_cntr_adr,j) for j in range(1, max(a_cntr_adr)+1)]
        a_subseqs_indic = [find_inds(a_cntr_indic,j) for j in range(1, max(a_cntr_indic)+1)]
        h_subseqs_adr = [find_inds(h_cntr_adr,j) for j in range(1, max(h_cntr_adr)+1)]
        h_subseqs_indic = [find_inds(h_cntr_indic,j) for j in range(1, max(h_cntr_indic)+1)]
        matches_adr += sum([1 for a in a_subseqs_adr if sum([1 for h in h_subseqs_adr if approx_match(a,h)]) > 0])
        matches_indic += sum([1 for a in a_subseqs_indic if sum([1 for h in h_subseqs_indic if approx_match(a,h)]) > 0])

    try:
        precision_adr = float(matches_adr) / float(auto_tags_adr)
    except ZeroDivisionError:
        precision_adr = 0.0
    try:
        precision_indic = float(matches_indic) / float(auto_tags_indic)
    except ZeroDivisionError:
        precision_indic = 0.0
    try:
        recall_adr = float(matches_adr) / float(hand_tags_adr)
    except ZeroDivisionError:
        recall_adr = 0.0
    try:
        recall_indic = float(matches_indic) / float(hand_tags_indic)
    except ZeroDivisionError:
        recall_indic = 0.0
    try:
        f1score_adr = 2*precision_adr*recall_adr/(precision_adr + recall_adr)
    except ZeroDivisionError:
        f1score_adr = 0.0
    try:
        f1score_indic = 2*precision_indic*recall_indic/(precision_indic+recall_indic)
    except ZeroDivisionError:
        f1score_indic = 0.0
    try:
        precision = float(matches_adr + matches_indic) / float(auto_tags_adr + auto_tags_indic)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = float(matches_adr + matches_indic) / float(hand_tags_adr + hand_tags_indic)
    except ZeroDivisionError:
        recall = 0.0
    try:
        f1score = 2*precision*recall/(precision+recall)
    except ZeroDivisionError:
        f1score = 0.0
    print('Approximate Matching Results:\n  ADR: Precision '+ str(precision_adr)+ ' Recall ' + str(recall_adr) + ' F1 ' + str(f1score_adr)
        + '\n  Indication: Precision ' + str(precision_indic) + ' Recall ' + str(recall_indic) + ' F1 ' + str(f1score_indic)
        + '\n  Overall: Precision ' + str(precision) + ' Recall ' + str(recall) + ' F1 ' +  str(f1score))
    return {'p':precision_adr, 'r':recall_adr, 'f1':f1score_adr}