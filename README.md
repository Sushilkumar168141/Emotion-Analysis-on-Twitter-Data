# Emotion-Analysis-on-Twitter-Data
### Steps:
1. Import libraries
```python
import twitter
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
import unicodedata
from wordcloud import WordCloud
from spacy.lang.en.stop_words import STOP_WORDS
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
import pprint

from sklearn.linear_model import SGDClassifier , LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
```
2. Authentication with Twitter credentials
```python
# These are sample credentials  you have to use your own credentials by creating  twitter  developer account
CONSUMER_KEY = 'Ri9pFOQqsdfasSDFdfdf12IZj5QCw26'
CONSUMER_SECRET = 'hRrAioGsLbwWHp0fAmrCNafsd1fADSF3mRYeAQ6OUZlQD36y1jej'
OAUTH_TOKEN = '1258017464174219269-y9B2adfa212f5YEHoSqi3qJ7RXGuMY'
OAUTH_TOKEN_SECRET = 'i8JW4em2QuH0eadFASDFsdfdsfdfdXSBDaafDsZJrmweYc79Z'
auth = twitter.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,
CONSUMER_KEY, CONSUMER_SECRET)
twitter_api = twitter.Twitter(auth=auth)
```
3. Preprocessing our dataset
```python
def preprocessing(text_and_label):
    text_and_label['hashtag_count'] = text_and_label['text'].apply(hashtag_count)
    text_and_label['mentions_count'] = text_and_label['text'].apply(mention_count)
    text_and_label['word_counts'] = text_and_label['text'].apply(lambda x : len(str(x).split()))
    text_and_label['average_word_len'] = text_and_label['text'].apply(get_average_word_len)
    text_and_label['text'] = text_and_label['text'].apply(cont_to_exp)
    text_and_label['stopword_count'] = text_and_label['text'].apply(stop_word_count)
    text_and_label['text'] = text_and_label['text'].apply(lambda x: ' '.join([t for t in x.split(' ') if t not in STOP_WORDS]))
    text_and_label['email_counts'] = text_and_label['text'].apply(get_email_count)
    text_and_label['text'] = text_and_label['text'].apply(remove_email)
    text_and_label['text'] = text_and_label['text'].apply(remove_url)
    text_and_label['text'] = text_and_label['text'].apply(lambda x : re.sub('RT','',x))
    #text_and_label['text'] = text_and_label['text'].apply(lambda x: re.sub('[^a-z A-Z 0-9-]+','',x))
    text_and_label['text'] = text_and_label['text'].apply(lambda x: " ".join(x.split(' ')))
    text_and_label['text'] = text_and_label['text'].apply(remove_accented_chars)
    text = ' '.join(text_and_label['text'])
    text = text.split(' ')
    freq = pd.Series(text).value_counts()
    x = ' '.join(text)
    wc = WordCloud(width = 800, height = 400).generate(x)
    plt.imshow(wc)
    plt.axis('off')
    plt.imshow(wc)
    plt.axis('off')
    return (text_and_label)
```
4. Training our model
```python
# storing models in a dictionary
clf={'SGD':sgd, 'LR':lr, 'SVM':svm, 'RFC':rfc}  

X, y = (df_bow, df_y)

# Defining  different  models
sgd=SGDClassifier(n_jobs=-1,random_state=2,max_iter=500)
lr=LogisticRegression(random_state=2,max_iter=2000)
svm=LinearSVC(random_state=2,max_iter=2000)
rfc=RandomForestClassifier(n_jobs=-1,random_state=2,n_estimators=200)

def classify(X,y):
    # Transforming  our dataset
    scaler=MinMaxScaler(feature_range=(0,1))
    X=scaler.fit_transform(X)
    
    # Splitting our dataset in training and test set
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
    
    # Train on different models
    for key in clf.keys():
        model=clf[key]
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        acc=accuracy_score(y_test,y_pred)
        print(key,"---->",acc)
        print(confusion_matrix(y_pred, y_test))
        print(classification_report(y_pred,y_test))

Classify(X, y)
```

6. Result  of different  models

|Model Name|Accuracy|Time taken|
|-------------|----------|-------------|
|Stochastic Gradient Descent | 65% | 55.6s |
|Logistic Regression | 66% | 7min 43s |
|Support Vector Machine|66% | 12.2 * 10^-6 s |
|Random Forest Classifier|65%|8.88 s|

8. Testing our model
```python
q = world_trends[0]['trends'][random.randint(0,len(world_trends[0]['trends']))]['name']
print("Trend : ",q)
tweets = twitter_api.search.tweets(q='IPL' + '-filter:retweets', lang='en' ,count=10, tweet_mode='extended')
query_tweets=[]
for i in range(len(tweets['statuses'])):
    query_tweets.append(tweets['statuses'][i]['full_text'])

# collecting tweets from another trends
q = world_trends[0]['trends'][random.randint(0,len(world_trends[0]['trends']))]['name']
print("Trend : ",q)
tweets = twitter_api.search.tweets(q=q + '-filter:retweets', lang='en' ,count=10, tweet_mode='extended')
for i in range(len(tweets['statuses'])):
    query_tweets.append(tweets['statuses'][i]['full_text'])


query_df = pd.DataFrame(query_tweets)
query_df.columns = ['text']
print(query_df)
new_df = query_df.copy(deep=True)
tweets_df = preprocessing(new_df)

text_counts = cv.transform(tweets_df['text'])
text_counts.toarray().shape
test = pd.DataFrame(text_counts.toarray(), columns = cv.get_feature_names())

result = svm.predict(test)
```
