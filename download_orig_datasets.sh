#!/usr/bin/env bash

# Download ASU Twitter ADR Dataset (v1.0) (Link last tested Sep 2016)
wget -O ./data/asu_tweets.zip http://diego.asu.edu/downloads/publications/ADRMine/download_tweets.zip 
unzip ./data/asu_tweets.zip -d ./data/asu_tweets
rm ./data/asu_tweets.zip

# Download Word2Vec Twitter Model (4.25GB, may take awhile) (Link last tested Sep 2016)
wget -O ./word2vec_twitter_model/word2vec_twitter_model.tar.gz http://yuca.test.iminds.be:8900/fgodin/downloads/word2vec_twitter_model.tar.gz
tar -xzvf ./word2vec_twitter_model/word2vec_twitter_model.tar.gz
rm ./word2vec_twitter_model/word2vec_twitter_model.tar.gz
