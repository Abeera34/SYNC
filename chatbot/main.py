import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

import pandas as pd

from tensorflow.keras.models import load_model

lt=WordNetLemmatizer()

with open('intents.json','r') as f:
   intents= json.load(f)

words=pickle.load(open('words.pkl','rb'))
classes=pickle.load(open('classes.pkl','rb'))
model=load_model('CollegeBuddy_model.h5')

# we get neumerical data from model we need to convert that into ans

def cleanup_sentence(sentence): #function to clean the sentence we get
   sentence_words=nltk.word_tokenize(sentence)
   sentence_words=[lt.lemmatize(word) for word in sentence_words]
   return sentence_words

def bag_of_words(sentence): #function to convert that sentence into neumerical list
   sentence_words=cleanup_sentence(sentence)
   bag=list(np.zeros(len(words)))
   for w in sentence_words:
      for i,word in enumerate(words):
         if word==w:
            bag[i]=1
   return np.array(bag)

def predict_class(sentence):
   bow=bag_of_words(sentence)
   res=model.predict(np.array([bow]))[0]
   threshold=0.5
   result=[[i,r] for i,r in enumerate(res) if r>threshold]

   result.sort(key=lambda x:x[1],reverse=True)
   return_list=[]

   for r in result:
      return_list.append({'intent': classes[r[0]],'probablity':str(r[1])})

   return return_list

def get_response(intents_list,intents_json):
   tag=intents_list[0]['intent']
   list_of_intents=intents_json['intents']
   for i in list_of_intents:
      if i['tag']==tag:
         result=random.choice(i['responses'])
         break
   return result

print("CollgeBuddy is Ready to Chat")

while True:
   messege=input("You:")
   ints=predict_class(messege)
   res=get_response(ints,intents)
   print("CollegeBuddy: ",res)



