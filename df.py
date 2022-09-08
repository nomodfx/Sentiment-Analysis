#TODO: 
#      Clean dataset slang
from time import time
import random, re, string
from nltk import text
import pandas as pd 
import matplotlib.pyplot as plt 
from nltk.tokenize import TweetTokenizer
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import classify
from nltk import NaiveBayesClassifier
from csv import reader
from nltk.tokenize.casual import TweetTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
STOP_WORDS = stopwords.words('english') 

#importing data
ds_raw = pd.read_csv('df_total.csv', encoding="UTF-8", header=None, sep=',')
ds_raw.columns = ["Review", "Rating"]
ds = ds_raw[['Review', 'Rating']] #only takes review & rating tags
ds.head()


#Filters sentiment 
positive_tweets = ds[ds['Rating'] == 4]
print(positive_tweets)#tests/filters output, delete
negative_tweets = ds[ds['Rating'] == 2]
print(negative_tweets)#tests/filters output, delete
neutral_tweets = ds[ds['Rating'] == 3]
print(neutral_tweets)#tests/filters output, delete

#Counts the total reviews for each tag
ds_pos = positive_tweets.iloc[:int(len(positive_tweets)/10)]
ds_neg = negative_tweets.iloc[:int(len(negative_tweets)/10)]
ds_neut = neutral_tweets.iloc[:int(len(neutral_tweets)/10)]
print(len(ds_pos), len(ds_neg), len(ds_neut))
ds = pd.concat([ds_pos, ds_neg, ds_neut])
len(ds)
print("data split")
print("\n")

#Tokenizer 
tk = TweetTokenizer(reduce_len=True)

data = []
X = ds['Review'].tolist()
Y = ds['Rating'].tolist()

for x, y in zip(X,Y):
    if y == 4:
        data.append((tk.tokenize(x), "Positive"))
    elif y == 3:
        data.append((tk.tokenize(x), "Neutral"))
    else:
        data.append((tk.tokenize(x), "Negative"))

data[:5]

#Label N/V/A removes inflectional endings
def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startsWith('NN'):
            pos = 'n' #noun
        elif tag.startswith('VB'):
            pos = 'v' #verb
        else:
            pos = 'a' #adjective
        lemmatize_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

#Removes common twitter slang
def remove_slang(token):
    if token == 'u':
        return 'you'
    if token == 'urs':
        return 'yours'
    if token == 'r':
        return 'are'
    if token == 'pls':
        return 'please'
    if token == 'plz':
        return 'please'
    if token == '2day':
        return 'today'
    if token == 'some1':
        return 'someone'
    if token == 'yrs':
        return 'years'
    if token == 'hrs':
        return 'hours'
    if token == 'mins':
        return 'minutes'
    if token == '4got':
        return 'forgot'
    if token == 'abt' and token == 'ab':
       return 'about'
    if token == 'adventuritter':
       return 'adventerous'
    if token == 'attwaction':
       return 'attraction'
    if token == 'b/c':
       return 'because'
    if token == 'b4':
       return 'before'
    if token == 'bfn':
       return 'bye for now'
    if token == 'cld':
       return 'could'
    if token == 'cre8':
       return 'create'
    if token == 'dm':
       return 'direct message'
    if token == 'fab':
       return 'fabulous'
    if token == 'f2f':
       return 'face to face'
    if token == 'ftl':
       return 'for the loss'
    if token == 'ftw':
       return 'for the win'
    if token == 'ic':
       return 'I see'
    if token == 'idk':
       return "I don't know"
    if token == 'imo':
       return 'in my opinion'
    if token == 'kk' and token == 'k':
       return 'ok'
    if token == 'mrt':
       return 'modified retweet'
    if token == 'mtf':
       return 'more to follow'
    if token == 'nts':
       return 'note to self'   
    return token
    
    
    
#cleaning dataset
def cleantokens(tweet_tokens):
    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens):

        #remove links
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        #remove mentions
        token = re.sub("(@[A-Za-z0-9_]+)","", token)
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)
        cleaned_token = remove_slang(token.lower())
        if cleaned_token not in string.punctuation and len(cleaned_token) > 2 and cleaned_token not in STOP_WORDS:
            cleaned_tokens.append(cleaned_token)
    return cleaned_tokens



def list_to_dict(cleaned_tokens):
    return dict([token, True] for token in cleaned_tokens)

cleaned_tokens_list = []

for tokens, label in data:
    cleaned_tokens_list.append((cleantokens(tokens), label))
 


final_data = []

for tokens, label in cleaned_tokens_list:
    final_data.append((list_to_dict(tokens), label))

def polarity_scores(tweet):
    sid_obj = SentimentIntensityAnalyzer()
    
    #detects polarity
    sentiment_dict = sid_obj.polarity_scores(tweet)
    
    print("Overall Sentiment: ",sentiment_dict)
    print(sentiment_dict['pos']*100,"% Positive")
    print(sentiment_dict['neg']*100,"% Negative")
    print(sentiment_dict['neu']*100,"% Neutral")
    
    print("Overall Rating:", end = " ")
    
    #decide if sentiment has multiple instances of pos, neg, neut
    if sentiment_dict['compound'] >= 0.05:
        print("Positive")
    elif sentiment_dict['compound'] <= -0.05:
        print("Negative")
    else:
        print("Neutral")

final_data[:5]
#determine accuracy compared to NLTK's
random.Random(140).shuffle(final_data)
trim_index = int(len(final_data) * 0.9)
train_data = final_data[:trim_index]
test_data = final_data[trim_index:]

classifier = NaiveBayesClassifier.train(train_data)


print('Accuracy on train data:', classify.accuracy(classifier, train_data))
print('Accuracy on test data:', classify.accuracy(classifier, test_data))
print(classifier.show_most_informative_features(21))


#custom_tweet = "I love league of legends so much." #positive statement
custom_tweet = "Computers exist" #neutral statement
#custom_tweet = u"Resident Evil Village is amazing: 😀!" #positive emoticon/emoji
#custom_tweet = u"Outriders was horrible imo: 😠!" #negative emoticon/emoji
custom_tokens = cleantokens(tk.tokenize(custom_tweet))

#prints the custom_tweet and whats been classified.
print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))
 
#checks sentiement polarity and what has been classified
print("\nTweet Sentiment Analysis: ")
polarity_scores(custom_tweet)   
 