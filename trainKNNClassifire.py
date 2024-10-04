#Training KNN Classifier on IMDB Dataset

#Traing Data
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