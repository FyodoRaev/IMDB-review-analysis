import sys
import numpy as np

def sigmoid(x):
  return 1/(1 + np.exp(-x))

def init_params_for_NLP(vocab , hidden_size ):
  weights_0_1 = 0.2*np.random.random((len(vocab),hidden_size)) - 0.1
  weights_1_2 = 0.2*np.random.random((hidden_size,1)) - 0.1
  return weights_0_1 , weights_1_2

def get_data():
  f = open('C:/Users/volor/Desktop/machine learning/reviews.txt')
  raw_reviews = f.readlines()
  f.close()
  f = open('C:/Users/volor/Desktop/machine learning/labels.txt')
  raw_labels = f.readlines()
  f.close()
  return raw_reviews , raw_labels

def create_vector_vocab(raw_reviews): # we create tokens list with all words used in reviews
  tokens = list(map(lambda x:set(x.split(" ")), raw_reviews )) #lambda is a python build-in literal that exist for creating one-time-use functions
  vocab = set() #cause set() do not have duplicats and we want to not have same words in our vector representation of words
  for sent in tokens:
    for word in sent:
      if len(word)>0:
        vocab.add(word)
  vocab = list(vocab)

  word2index = {}
  for i,word in enumerate(vocab): #adding indeces for all words
    word2index[word]=i 
  return tokens , vocab , word2index 
