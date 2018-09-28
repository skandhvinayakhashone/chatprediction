"""
Created on Sat Jul  7 12:37:29 2018
Property Of Institute Of Analytics
@author: skandhvinayak
"""


import math
import textblob
from textblob import TextBlob as tb
import pandas as pd
import numpy as np
import sklearn
from sklearn.cross_validation import train_test_split

#cleaning chatdata
chatdataframe=pd.read_csv("/Users/skandhvinayak/Downloads/chat.csv",sep=',')
del chatdataframe['intent']
chatdataframe.to_csv('/Users/skandhvinayak/Downloads/newchat.txt', sep=',',index=False,header=False)
chatdataframe=pd.read_csv("/Users/skandhvinayak/Downloads/chat.csv",sep=',')
del chatdataframe['qn']
chatdataframe.to_csv('/Users/skandhvinayak/Downloads/newchat2.txt', sep=',',index=False,header=False)

bloblist=[]
text3=[]
rowval=[]
colval=[]
textmas=[]

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

def liney():
    def send(st):
        bloblist.append(st)

    with open("/Users/skandhvinayak/Downloads/newchat.txt","r") as t1 :  
        line = t1.readline()
        line=line.replace("\"",' ')
        line=line.replace("\'",' ')
        line=tb(line)
        send(line)
        cnt = 0
        while line:
           line = t1.readline()
           line=line.replace("\"",' ')
           line=line.replace("\'",' ')
           line=tb(line)
           send(line)
           cnt += 1
                        
def unique():
    t2=open("/Users/skandhvinayak/Downloads/newchat.txt","r") 
    text=t2.read()
    text=text.replace("\"",' ')
    text=text.replace("\'",' ')
    text=text.replace("?",' ')
    text=text.split()
    output=set(text)
    for i in output:
        rowval.append(i)
    
def intent_array():    
    def send(st):
        text3.append(st)
        
    with open("/Users/skandhvinayak/Downloads/newchat2.txt","r") as t3 :  
        line = t3.readline()
        line=line.replace("\"",' ')
        line=line.replace("\'",' ')
        send(line)
        cnt = 0
        while line:
           line = t3.readline()
           line=line.replace("\"",' ')
           line=line.replace("\'",' ')
           send(line)
           cnt += 1

liney()
unique()
intent_array()

for i, blob in enumerate(bloblist):
    scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    n=("document {}".format(i+1))
    colval.append(n)

#creating dataFrame
df=pd.DataFrame(index=colval,columns=rowval)
for i in text3:
    t=i.rstrip()
    textmas.append(t)
se = pd.Series(textmas)
df['intent']=se.values

headerlist=list(df.columns.values)

for i, blob in enumerate(bloblist):
    
    scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:10]:
        if word in headerlist:
            
            df.at['document {}'.format(i+1),'{}'.format(word)]=score
        else:
            df.at['document {}'.format(i+1),'{}'.format(word)]= float(0)
     
for i in headerlist:
    df['{}'.format(i)]=df['{}'.format(i)].fillna(0)
df=df.drop('document 55')

#writing dataFrame to a csv
df.to_csv('/Users/skandhvinayak/Downloads/dataframe.csv', sep=',',index=False)
y=df['intent']
headerlist.remove('intent')
h=headerlist

#seperation test and train data
x=df[h]
y = pd.DataFrame(y.values)
x = pd.DataFrame(x.values)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 2000)

#Applying GaussianNB
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#Predicting The Test Values
y_pred = classifier.predict(X_test)

#Creating Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Getting accuracy
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test, y_pred)
print(acc)
#Accuracy 78.5%

