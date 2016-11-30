import numpy

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from operator import itemgetter
y_train = []
y_test = []
i=0
# taking the sentiment values from text file
with open('sentiments.txt') as f:
    for line in f:
	if(i<775000):
		y_train.append(int(line))
	else:
		y_test.append(int(line))
	i=i+1
	

print("SENTIMENT RECORDED") 
# creating the hash table for the word indexes
hash = {}
temp = []
with open('vocab.txt') as f:
    for line in f:
        key, value = line.strip().split(' ', 1)
        hash[key] = int(value)
print("Word-index hash table created)
#creating word-vector hash table
hash2 = {}
with open('vectors.txt') as f:
    for line in f:
        key, value = line.strip().split(' ', 1)
	temp = (value.split(' '))
	x = numpy.array(temp)
	y=x.astype(numpy.float)
        hash2[key] = y
print("Word-Vectors hash table created")
#creating tweets' training and test sets
X_train = []
X_test = []
i=0
with open('text8') as f:
    for line in f:
	
	if(i<775000):
		X_train.append([])
	else:
		X_test.append([])
	for word in line.split():
		if(i<775000):
			X_train[i].append(hash[word])
		else:
			X_test[i-775000].append(hash[word])
	i=i+1
	if(i==273576):break
print("Testing and training data prepared for entry")
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
vocab_dim = 50
n_symbols = len(hash) + 1 # adding 1 to account for 0th index (for masking)


embedding_matrix = numpy.zeros((n_symbols + 1, vocab_dim))
for word, i in hash.items():
    embedding_vector = hash2.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


model = Sequential()
model.add(Embedding(input_dim=n_symbols+1,output_dim=50,input_length = 30, weights=[embedding_matrix]))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
X_train = sequence.pad_sequences(X_train, maxlen=30)
X_test = sequence.pad_sequences(X_test, maxlen=30)



print("\n\n done with the creation of layers")
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=2, batch_size=128, verbose=1)
# Final evaluation of the model
print("Done fitting")
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


