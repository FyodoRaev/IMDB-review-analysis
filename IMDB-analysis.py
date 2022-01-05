import sys
import numpy as np
from functions import get_data
from functions import sigmoid
from functions import init_params_for_NLP
from functions import create_vector_vocab
np.random.seed(1) #The random.seed() method sets the initial conditions for a random number generator.



raw_reviews , raw_labels = get_data()
tokens , vocab , word2index = create_vector_vocab(raw_reviews)



input_dataset = list()
for sent in tokens:
  sent_indices = list()
  for word in sent:
    try:
      sent_indices.append(word2index[word])
    except:
      ""
  input_dataset.append(list(set(sent_indices)))

target_dataset = list()
for label in raw_labels:
  if label == 'positive\n':
    target_dataset.append(1)
  else:
    target_dataset.append(0)



alpha, iterations = (0.01 , 2)
hidden_size = 100

weights_0_1 , weights_1_2 = init_params_for_NLP(vocab , hidden_size )

correct,total = (0,0)
for iter in range(iterations):

  for i in range(len(input_dataset)-1000): # we will teach network on first 24000 reviews cause we want to check it's results on last thousand
    x,y = [input_dataset[i], target_dataset[i]] # x is input information and y is what we want to get,target variable
    layer_1 = sigmoid(np.sum(weights_0_1[x],axis=0))
    layer_2 = sigmoid(np.dot(layer_1,weights_1_2)) # <- a simply connected linear layer

    layer_2_delta = layer_2 - y 
    layer_1_delta = layer_2_delta.dot(weights_1_2.T)

    weights_0_1[x] -= layer_1_delta * alpha
    weights_1_2 -= np.outer(layer_1, layer_2_delta) * alpha #simple backward propagation , we multiple values on weights

    if np.abs(layer_2_delta)< 0.5:
        correct += 1
    total += 1
    if i%10 == 9:
      progress = str(i/float(len(input_dataset)))
      print('\rIter:' + str(iter) + ' Progress:'+ progress[2:4] + "." + progress[4:6] + " % Training accuracy: " + str(correct/float(total)) + " %")


