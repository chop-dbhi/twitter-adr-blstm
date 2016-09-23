CHOP ADHD ADR Twitter Supplement dataset

This directory contains Tweets mentioning ADHD medications that have been annotated for adverse drug reactions. The annotated tweets were collected via the Twitter streaming API using the search terms contained in ./twitter_drug_search_terms

The data set is meant to act as a supplement to the original ASU Twitter ADR Dataset v1.0 [1] and has been developed
using the same format for compatibility.

The tweets were annotated under the same guidelines used in the original work.

NOTE

This dataset release contains only a portion of the CHOP supplemental data used in the publication. When we collected and annotated the tweets in 2015, our supplemental dataset contained 201 unique tweets. At the time of publication, only 87 of these were available for download from Twitter (due to deleted user accounts and restrictions on viewing user history). If you need the full supplemental dataset, please contact the authors directly.

By downloading the Twitter data you agree to follow the Twitter terms of service (https://twitter.com/tos).

DATA

For compatibility, we have followed the same format as the ASU Twitter ADR Dataset authors. Their formatting information is copied here. 

The Twitter data set is divided into train and test sets. For each set the information about the tweets and the annotations are saved in separate files: Tweet ID files and annotation files .
Tweet ID Files: train_tweet_ids.tsv, test_tweet_ids.tsv

These files contain tab separated information about tweet IDs, user IDs and text IDs as shown in the example below. 

351772771174449152	632397611	10238

The tweet ID and user ID can be used by the Twitter API for downloading the tweets. The text ID links the tweet to the corresponding annotations in the annotation file.  


Annotation Files: train_tweet_annotations.tsv, test_tweet_annotations.tsv

These files contain tab separated information about the details of the annotations including: text ID, start offset, end offset, semantic type, annotated text, related drug and target drug. The following line is an example annotation line. Please note that the related drug is the (properly spelled) drug that was used as a keyword in Twitter search query and the target drug is the drug that the current annotation (ADR or Indication) is targeting. Target drug can be different from the related drug in cases where there are more than one drug mentions in a tweet. 

10238	13	34	ADR	Restless Leg Syndrome	fluoxetine	fluoxetine



DOWNLOAD
The individual tweets can be obtained via the url:
 http://twitter.com/[userid]/status/[tweetid]
For example, for the sample line from the test_tweet_ids, the tweet can be accessed via the following url:

https://twitter.com/632397611/status/342043859523600384

In order to make the obtaining of the associated tweets easier for researchers, we provide a simple python script:

- download_tweets.py

This script requires python to be installed along with the beautifulsoup4 package.

beautifulsoup4 can be installed easily via easy_install:

easy_install beautifulsoup4

To run the script, please use the following command:

python download_tweets.py [input_filename] > [output_filename]



REFERENCES

[1] Nikfarjam A, Sarker A, O’Connor K, et al. Pharmacovigilance from social media: mining adverse drug reaction
    mentions using sequence labeling with word embedding cluster features. J Am Med Inform Assoc 2015;22(3):671-81.

CITATION

If you use this data in your work, please cite the following paper:

CONTACT

If you require access to the full annotated supplemental dataset, please contact one of the authors below:

Anne Cocos
acocos@seas.upenn.edu

Aaron Masino
masinoa@email.chop.edu