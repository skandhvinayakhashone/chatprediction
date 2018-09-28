#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 12:37:29 2018
Property Of Institute Of Analysis
@author: skandhvinayak
"""

#importing libraries
import numpy as np
import pandas as pd
from numpy  import array


#importing dataset
dataset = pd.read_csv('/Users/skandhvinayak/Downloads/chat.csv', sep=',')

#checking for null values
dataset.isnull().sum()

#checking the number of intents
dataset.intent.value_counts()

#splitting the dataset
X = dataset.iloc[:,0].values
y = dataset.iloc[:,1].values

#converting numpy nd array to list
newlist=np.array(X).tolist()

#removing the stop words
from nltk.corpus import stopwords
cachedStopWords = stopwords.words("english")

for x in range(0,len(newlist)):
    newlist[x] = ' '.join([word for word in newlist[x].split() if word not in cachedStopWords])

for x in range(0,len(newlist)):
    newlist[x]='{}'.format(newlist[x]).replace('I','').replace('may','')

from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer 
stem = PorterStemmer()

def steming(input_text):
    words = input_text.split()
    stem_free_words = [stem.stem(word) for word in words] 
    stem_free_text = " ".join(stem_free_words)
    return stem_free_text

for i in range(0,len(newlist)):
    newlist[i]=steming(newlist[i])

#converting back to numpy nd array
X = array( newlist )

#converting factors to objects
y = y.astype('object')

#implementing BOW
from sklearn.feature_extraction.text import CountVectorizer
bag=CountVectorizer(max_features=5000)
print(bag)
X = bag.fit_transform(X).toarray()


#encoding the levels in y
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#importing keras and converting y to catagorical
import keras
from keras.utils import np_utils
y = np_utils.to_categorical(y)

#splitting dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 20000)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# Import Keras libraries 
from keras.models import Sequential
from keras.layers import Dense



# ANN
classifier = Sequential()

classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 35))

classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 5, epochs = 20) # Lesser no of epochs - Basic Model

# Prediction
y_pred = classifier.predict(X_test)

maxi = y_pred.max(axis=1)


for i in range(len(y_pred)):
    for j in range(10):
        if y_pred[i,j] == maxi[i]:
           y_pred[i,j] = 1
        else:
               y_pred[i,j] = 0
     

# Accuracy    
crt_values = (y_pred == y_test).sum()
wrong_values = (y_pred != y_test).sum()
total = crt_values+wrong_values
result = crt_values/total
print(result) 
# 88.5% accuracy(50 epochs)
#85.7% accurracy(20 epochs)

