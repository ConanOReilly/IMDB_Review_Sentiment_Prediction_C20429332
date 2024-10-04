#Training KNN Classifier on IMDB Dataset

#Traing Data
import pandas as pd

train_df = pd.read_csv('https://raw.githubusercontent.com/aws-samples/aws-machine-learning-university-accelerated-nlp/master/data/final_project/imdb_train.csv', header=0)
print('The shape of the dataset is:', train_df.shape)

train_df["label"].value_counts()

