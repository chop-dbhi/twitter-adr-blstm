#!/usr/bin/env python
######################################
#
# download_tweets.py
#
# Author: Anne Cocos
# Date:   May, 2016
#
# Given Tweet ID and Annotation files from the
# original ASU Twitter ADR Dataset and the CHOP
# ADHD ADR Dataset Supplement, download the
# original text for each from Twitter. Then
# combine tweets with annotations.
#
# Contains code originally published with the
# ASU Twitter ADR Dataset.
######################################
import os
import sys
import urllib
import re
import json
import socket
import csv

from bs4 import BeautifulSoup

from settings import SETTINGS

def check_files(basedir):
    for f in ['test_tweet_annotations.tsv', 'test_tweet_ids.tsv',
              'train_tweet_annotations.tsv', 'train_tweet_ids.tsv']:
        if not os.path.isfile(os.path.join(basedir, f)):
            sys.stderr.write('Cannot locate necessary file %s.\n'
                             'Please ensure you run the script `download_orig_datasets.sh` '
                             'before this one, and make sure the dataset paths in settings.py'
                             'point to the actual dataset locations.\n')
            return False
    return True

def get_tweet_text(uid, tid):
    tweet = None
    text = None

    f = urllib.urlopen('http://twitter.com/'+str(uid)+'/status/'+str(tid))
    html = f.read().replace("</html>", "") + "</html>"
    soup = BeautifulSoup(html)
    jstt   = soup.find_all("p", "js-tweet-text")
    tweets = list(set([x.get_text() for x in jstt]))

    if len(tweets) > 1:
        other_tweets = []
        cont = soup.find_all("div", "content")
        for i in cont:
            o_t = i.find_all("p", "js-tweet-text")
            other_text = list(set([x.get_text() for x in o_t]))
            other_tweets += other_text
        tweets = list(set(tweets) - set(other_tweets))

    text = tweets[0]
    for j in soup.find_all("input", "json-data", id="init-data"):
        js = json.loads(j['value'])
        if js.has_key("embedData"):
            tweet = js["embedData"]["status"]
            text = js["embedData"]["status"]["text"]
            break

    if tweet is not None and tweet["id_str"] != tid:
        text = "This tweet has been removed or is not available"
    text = re.sub(r'\s+', ' ', text)
    return text


def download_tweets(filename):
    ''' Downloads tweets for each tweet internalID in input filename, returns
    dict of {internalID -> text}
    :param filename: str
    :return:
    '''
    item_dict = {}
    for line in open(filename, 'rU'):
        fields = line.strip().split('\t')
        tweetid, userid, internalid = fields
        text = item_dict.get(internalid, None)
        if not text:
            try:
                text = get_tweet_text(userid, tweetid)
                item_dict[internalid] = text
            except:
                sys.stderr.write("Error downloading tweet %s\n" % tweetid)
                continue
    return item_dict


def download_dataset(basedir):
    d = {'train': {}, 'test': {}}
    for t in d:
        fname = os.path.join(basedir, '%s_tweet_ids.tsv' % t)
        d[t] = download_tweets(fname)
    return d


def resolve_tweets(tid2text, annofile, outfile):
    # We're treating the DRUG semantic type in ASU data as 'NEG' for ADR or Indication
    clslookup = {'ADR': 'ADR', 'Indication': 'Indication', 'Drug':'NEG', 'NEG':'NEG', 'Beneficial': 'NEG'}
    headers = ['id', 'start', 'end', 'semantic_type', 'span', 'reldrug','tgtdrug','text']
    with open(outfile, 'w') as fout:
        w = csv.writer(fout)
        w.writerow(headers)
        with open(annofile, 'rU') as fin:
            for line in fin:
                entry = line.strip().split('\t')
                entry[3] = clslookup[entry[3]]
                id = entry[0]
                if id in tid2text:
                    w.writerow(entry + [tid2text[id].strip().encode('utf8')])


def resolve_dataset(dname, basedir, tweetdict):
    for t in ['train', 'test']:
        annofile = os.path.join(basedir, '%s_tweet_annotations.tsv' % t)
        outfile = SETTINGS['SEQLAB_DATA_DIR']+'/raw/%s/%s_fullanno_%s' % (t, dname, t)
        try:
            os.makedirs(os.path.dirname(outfile))
        except:
            pass
        resolve_tweets(tweetdict[t], annofile, outfile)


if __name__=="__main__":
    chop_basedir = SETTINGS['CHOP_DATASET']
    asu_basedir = SETTINGS['ASU_DATASET']

    anno_basedir = SETTINGS['SEQLAB_DATA_DIR']+'/raw'
    try:
        os.makedirs(anno_basedir)
    except:
        pass

    for dname, basedir in zip(('chop', 'asu'), (chop_basedir, asu_basedir)):

        # Check to make sure data is there
        if not check_files(basedir):
            exit(0)

        # Download tweet text
        socket.setdefaulttimeout(20)
        tweet_text = download_dataset(basedir)

        resolve_dataset(dname, basedir, tweet_text)
