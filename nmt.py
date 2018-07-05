import keras
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
import nltk
import numpy as np 
import codecs
from keras.layers import LSTM , RepeatVector,Dense, TimeDistributed, Activation 
from keras.layers import Flatten 
from keras.layers.embeddings import Embedding
import operator 
vocab_size = 500
seq_length= 15 
filename1 = "strain.en"
filename2 = "strain.de"

#steps for preprocessing 
# prepare a dictionary with all the words including , ! etc 
# dict in the form word: frequency 
# now sort the words and put them according to their values and 
# place the words in a list 
# from this i can prepare my list of list of ids 
# im using nltk tokenizer to output list of list of word including , ! etc
# and padding them with zeros , also truncating 
# for source language
# creating list of list of words including , ! etc 
lines = []
with open(filename1) as file:
    for line in file:
        line = line.strip()  # gets rid of the newline character at the end of each sentence
        lines2 = nltk.word_tokenize(line)
        lines.append(lines2)
file.close()
#print lines
dict = {}
#creating the dict 
for sent in lines:
    for word in sent:
        if word.lower() in dict:
            dict[word.lower()] = dict[word.lower()] + 1
        else:
            dict[word.lower()]= 1

sorted_dict = sorted(dict.items(), key=operator.itemgetter(1), reverse = True)
#print sorted_dict[0][0]

out_list = []
new_dict = {}
#creating a new dictionary with top words 
for i,items in enumerate(sorted_dict):
    if i<vocab_size:
        new_dict[sorted_dict[i][0]]= i+1

#print new_dict
#create the list of list of ids 
for a,sentences in enumerate(lines):
    #print sentences
    #break
    in_list = []
    for b,words in enumerate(sentences):
        #print words 
        if words.lower() in new_dict:
            in_list.append(new_dict[words.lower()])
            #print in_list
        else:
            in_list.append(vocab_size+1)
            #print in_list 
            #break 
    out_list.append(in_list)
#print out_list 

#padding
x = pad_sequences(out_list, maxlen=seq_length,dtype= 'int32',padding='post', truncating= 'post', value= 0.0)
# need to reverse the source sequences 
pass
# splitting data 
cutoff = int(0.7*len(x)) 
x_train = x[:cutoff]
x_test = x[cutoff:]


#for target language  
# creating list of list of words including , ! etc 
lines_tar = []
with codecs.open(filename2, 'r', encoding='utf-8') as file:
#with open(filename1) as file:
    for line_tar in file:
        line_tar = line_tar.strip()  # gets rid of the newline character at the end of each sentence
        lines2_tar = nltk.word_tokenize(line_tar)
        lines_tar.append(lines2_tar)
file.close()
#print lines_tar
dict_tar= {}
#creating the dict 
for sent_tar in lines_tar:
    for word_tar in sent_tar:
        if word_tar.lower() in dict_tar:
            dict_tar[word_tar.lower()] = dict_tar[word_tar.lower()] + 1
        else:
            dict_tar[word_tar.lower()]= 1
#print dict_tar
sorted_dict_tar = sorted(dict_tar.items(), key=operator.itemgetter(1), reverse = True)
#print sorted_dict_tar

out_list_tar = []
new_dict_tar = {}
#creating a new dictionary with top words 
for i,items in enumerate(sorted_dict_tar):
    if i<vocab_size:
        new_dict_tar[sorted_dict_tar[i][0]]= i+1

#print new_dict_tar
#create the list of list of ids 
for a,sentences in enumerate(lines_tar):
    #print sentences
    #break
    in_list_tar = []
    for b,words in enumerate(sentences):
        #print words 
        if words.lower() in new_dict_tar:
            in_list_tar.append(new_dict_tar[words.lower()])
            #print in_list
        else:
            in_list_tar.append(vocab_size+1) # cuz ids should be continuous :w
            #print in_list 
            #break 
    out_list_tar.append(in_list_tar)
#print out_list_tar

#padding
y = pad_sequences(out_list_tar, maxlen=seq_length,dtype= 'int32',padding='post', truncating= 'post', value= 0.0)
# need not reverse the target sequences 

# splitting data 
#cutoff = int(0.7*len(x)) 
y_train = y[:cutoff]
y_test = y[cutoff:]
print y_test,'\n'
#print// x_test[0][0]
#x_train = np.array(x_train)
#y_train = seq_lengthnp.array(y_train)
#creating the model and embedding layer
model = Sequential()
#print model.input_shape 
# ids should be continuous...can give trained weights through weights arguments
model.add(Embedding(vocab_size+2,16,embeddings_initializer="uniform", input_length=seq_length))
#print model.input_shape 
model.add(LSTM(128,return_sequences = False , stateful = False)) # encoder lstm 
# adding repeact vector which repeats the final state of encoder , this 
# acts as imput for every lstm in decoder 
model.add(RepeatVector(seq_length))
# adding lstm layer and taking output at each time step unlike before 
model.add(LSTM(128, return_sequences=True, stateful = False)) # decoder lstm
# apparently need to add flatten when going from a rnn/lstm .ie 3D layer 
# to a 2D dense layer
#model.add(Flatten())
#adding a time-distributed dense layer . there is a dense 
# layer on top of every lstm 
model.add(TimeDistributed(Dense(vocab_size+2, activation='softmax')))
#model.add(TimeDistributed(Activation('softmax'))) # applying softmax on them
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model.summary()
print(y_train)
print(y_train.shape)
model.fit(x_train,y_train,batch_size=20 , epochs = 3)
# Final evaluation of the model
# x and y are in the form the numpy array ...list of list 
scores = model.evaluate(x_test, y_test, verbose=0)
print 'accuracy : ',scores[1]*100
#print len(x_test)
#print model.input_shape 
#To Do 
# can try reversing source seq
# can try bilstm 
# can try stateful = True 
# have a doubt in dense layer line
# doubt about Flatten()
# we are giving a fixed output seq length as well in decoder 
# but what if we can makeit terminate by itself when it produces <end> token 









