#Analysing IMDB Dataset using KNN Classifier

#Importing Dataset
import pandas as pd

train_df = pd.read_csv('https://raw.githubusercontent.com/aws-samples/aws-machine-learning-university-accelerated-nlp/master/data/final_project/imdb_train.csv', header=0)
print('The shape of the dataset is:', train_df.shape)

#Count entities in dataset
train_df["label"].value_counts()

#Count missing values
print(train_df.isna().sum())

#Text Processing
import nltk

nltk.download('punkt')
nltk.download('stopwords')

import nltk, re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

stop = stopwords.words('english')

excluding = ['against', 'not', 'don', "don't",'ain', 'aren', "aren't", 'couldn', "couldn't",
             'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
             'haven', "haven't", 'isn', "isn't", 'mightn', "mightn't", 'mustn', "mustn't",
             'needn', "needn't",'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
             "weren't", 'won', "won't", 'wouldn', "wouldn't"]

stop_words = [word for word in stop if word not in excluding]

snow = SnowballStemmer('english')

def process_text(texts):
    final_text_list=[]
    for sent in texts:

        if isinstance(sent, str) == False:
            sent = ""

        filtered_sentence=[]

        sent = sent.lower() 
        sent = sent.strip() 
        sent = re.sub('\s+', ' ', sent) 
        sent = re.compile('<.*?>').sub('', sent) 

        for w in word_tokenize(sent):

            if(not w.isnumeric()) and (len(w)>2) and (w not in stop_words):

                filtered_sentence.append(snow.stem(w))
        final_string = " ".join(filtered_sentence) 

        final_text_list.append(final_string)

    return final_text_list

#Training - Validation Split
from sklearn.model_selection import train_test_split

X=train_df[["text"]]
Y=train_df["label"]
X_train, X_val, y_train, y_val = train_test_split(X,
                                                  Y,
                                                  test_size=0.10,
                                                  shuffle=True,
                                                  random_state=324
                                                 )

print("Processing the text fields")
train_text_list = process_text(X_train["text"].tolist())
val_text_list = process_text(X_val["text"].tolist())

#Processing Data with Pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier


import gensim
from gensim.models import Word2Vec

w2v = gensim.models.Word2Vec()
pipeline = Pipeline([
    ('text_vect', CountVectorizer(binary=True,
    #( 'text_vect', TfidfVectorizer(use_idf=True,
                                  max_features=10)),
    ('knn', KNeighborsClassifier())
                                ])

from sklearn import set_config
set_config(display='diagram')
pipeline

#Train Classifier
X_train = train_text_list
X_val = val_text_list

pipeline.fit(X_train, y_train.values)

#Test Classifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

val_predictions = pipeline.predict(X_val)
print(confusion_matrix(y_val.values, val_predictions))
print(classification_report(y_val.values, val_predictions))
print("Accuracy (validation):", accuracy_score(y_val.values, val_predictions))