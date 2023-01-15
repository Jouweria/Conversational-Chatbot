
#Importing packages for preprocessing
import nltk
import numpy as np
import requests
from numpy import ndarray

response = requests.get("https://api.covidtracking.com/v1/us/current.json")
data = response.json()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk import WordNetLemmatizer
import numpy as np
import pickle
import string
import random
import re
import json  # Json documents have a strict format

#importing ML models
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Activation, Dense



# Reading files and converting files
Genesis_brain = json.loads(open('Genesis_brain.json').read())


# Text Preprocessing
# Function for removing punctuations using regex
def remove_punct(s):
    pattern = "[^\w\s]"
    s = re.sub(pattern, '', s)
    return (str(s))


words = [] #to create a bag of words
new_words = []
classes = []
documents = []



#Tokenization for every word in
for item in Genesis_brain['Questions']:
    for pattern in item['patterns']:
        word_list = nltk.word_tokenize(remove_punct(pattern))
        words.extend(word_list)
        documents.append((word_list,item['tag']))
        if item["tag"] not in classes:
            classes.append(item['tag'])


# Lemmitization
#For every word in words:
lemm = WordNetLemmatizer()
words = [lemm.lemmatize(word) for word in words]

words = sorted(set(words))  # set removes duplicates and sorted sorts in order
classes = sorted(set(classes))



# Save files for words and classes
with open('words.pkl','wb') as f:
    pickle.dump(words,f)  #wb means write in binary to avoid python from encoding the file
with open('classes.pkl','wb') as r:
    pickle.dump(classes,r)

#Create a bag of words
#One-Hot Encoding (Feature vectorization of every word)

training  = []
output_row = []

for document in documents:
    bag = []
    i = 0
    word_patterns = document[i]
    word_patterns = [word.lower() for word in word_patterns] #turn it into lower case words
    for word in words: #from the file "words.pkl"
        bag.append(1) if word in word_patterns else bag.append(0)

#create a training set

    output_row = list([0] * len(classes))
    output_row[classes.index(document[1])] = 1# I don't understand this?
    training.append([bag,output_row])
    i+=1
print(documents)
print(classes)
print(bag)
print(output_row)

random.shuffle(training)
training: ndarray = np.array(training)

train_x = list(training[:,0])#feature vector for every word
train_y = list(training[:,1])#the class it belongs to

#Now to train the model using neural networks (honestly didn't understand any of this)

model = Sequential()
model.add(Dense(128, input_shape = (len(train_x[0]),), activation = 'relu'))
model.add(Dropout(0,5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0,5))
model.add(Dense(len(train_y[0]), activation = 'softmax'))

sgd = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs = 200,batch_size = 5, verbose = 1)
model.save('chatbotmodel.h5', hist)
print('Done')

