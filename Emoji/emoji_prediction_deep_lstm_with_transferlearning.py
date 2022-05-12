'''
Use deep LSTM to predict emoji
'''

import os
os.chdir(r'C:\all\Desktop\Course Slides\TextMining\2022spring\11 text classification with deep learning\sentiment_analysis_with_deeplearning')

import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from tensorflow.keras.optimizers import Adam
import emoji
import csv

# step1: define all helper functions
def convert_to_one_hot(Y, C):
    '''
    arg:
        C: number of classes
        Y: 1-d np array, showing the class label such as [5,3,1,2,0...]
    return:
        one-hot encoded Y
    '''
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

def read_csv(filename):
    '''
    used to read first column text, second column label csv files
    '''
    phrase = []
    emoji = []

    with open (filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            phrase.append(row[0])
            emoji.append(row[1])

    X = np.asarray(phrase)
    Y = np.asarray(emoji, dtype=int)

    return X, Y

# create a mapping of the dataset labels to the emoji
emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}

def label_to_emoji(label):
    """
    Use the emoji library to converts a label (int or string) into the corresponding emoji code (string) ready to be printed
    """
    return emoji.emojize(emoji_dictionary[str(label)], language='alias')

def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding="utf8") as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

# step2: preprocess the train/test dataset
X_train, Y_train = read_csv('train_emoji.csv')
X_test, Y_test = read_csv('test_emoji.csv')
maxLen = len(max(X_train, key=len).split())

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')

def sentences_to_indices(X, word_to_index, max_len):
    """
    args:
        X: array of sentences (strings), of shape (m, 1)
        word_to_index: a dictionary containing the each word mapped to its index
        max_len: maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    returns:
    X_indices: array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    
    m = X.shape[0] # number of training examples
    
    # Initialize X_indices as a numpy matrix of zeros and the correct shape
    X_indices = np.zeros((m, max_len))
    
    for i in range(m):                               # loop over training examples
        
        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words =X[i].lower().split()
        
        # Loop over the words of sentence_words
        for index_w,w in enumerate(sentence_words):
            X_indices[i, index_w] = word_to_index[w]

    return X_indices

X1 = np.array(["funny lol", "lets play baseball", "food is ready for you"])
X1_indices = sentences_to_indices(X1,word_to_index, max_len = 5)
print("X1 =", X1)
print("X1_indices =\n", X1_indices)

# step3: define a non-trainable embedding layer (a computational graph)
def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a none trainable Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    We can acutally use pretrained models in your own model
    This is called transfer learning
    
    args:
    word_to_vec_map: dictionary mapping words to their GloVe vector representation.
    word_to_index: dictionary mapping from words to their indices in the vocabulary (400,000 words)
    
    returns:
    embedding_layer -- pretrained layer Keras instance
    """
    
    vocab_len = len(word_to_index) + 1 # adding 1 to accommodate unrecorded words. If a word is not in vocab, use this additional indice to represent
    emb_dim = word_to_vec_map["cucumber"].shape[0] # define dimensionality of your GloVe word vectors (= 50)
    
    # Step 1
    # Initialize the embedding matrix as a numpy array of zeros.
    # See instructions above to choose the correct shape.
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    # Step 2
    # load the pretrained embedding matrix to fill the emb_matrix
    # Set each row "idx" of the embedding matrix to be 
    # the word vector representation of the idx'th word of the vocabulary
    for word, idx in word_to_index.items():
        emb_matrix[idx, :] = word_to_vec_map[word]

    # Step 3
    # Define Keras embedding layer with the correct input and output sizes
    # Make it trainable, so that we can adapt the pretrained embedding layer to our dataset
    # if trainable is False, this will be exactly the same as the previous example, where we first use the embedding matrix to convert dataset into vector representation.
    embedding_layer = Embedding(vocab_len,emb_dim,trainable=True)

    # Step 4
    # Build the embedding layer, it is required before setting the weights of the embedding layer. 
    embedding_layer.build((None,)) # Do not modify the "None". It just sets where to put the batch size in the computational graph
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer


embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3]) # first layer, 1st word, the 3rd dimension of the embedding.. There is only one layer

# step4: define and train deep learning model
def model(input_shape, word_to_vec_map, word_to_index):
    """
    args:
        input_shape: shape of the input, (max_len,)
        word_to_vec_map: dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
        word_to_index: dictionary mapping from words to their indices in the vocabulary

    returns:
        model: a model instance in Keras
    """
    
    # Define sentence_indices as the input of the graph.
    # It should be of shape input_shape
    sentence_indices = Input(shape=input_shape) # shape is (m,max_len)
    
    # Create the embedding layer pretrained with GloVe Vectors
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    # Propagate sentence_indices through your embedding layer
    embeddings = embedding_layer(sentence_indices) # shape is (,max_len,embedding length)  
    
    # check the keras_lstm_instruction.py before moving on.
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden states
    # The returned output should be a batch of sequences.
    X = LSTM(units=128, return_sequences=True)(embeddings) # shape is (,max_len,128), max_len is like the number of time steps
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X) # The Dropout layer randomly sets numbers to 0 with a frequency of rate at each step during training time, which helps prevent overfitting.  shape is still (,max_len,128)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # The returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(units=128,return_sequences=False)(X) # shape is (,128)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X) # shape is still (,128)
    # Propagate hidden state through a Dense layer with 5 units
    y = Dense(5, activation = "softmax")(X) # shape (,5)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=y)
    
    return model

model = model((maxLen,), word_to_vec_map, word_to_index)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = convert_to_one_hot(Y_train, C = 5) # the Y variable shape must match the output shape defined in the model_V2 function (,5)
model.fit(X_train_indices, Y_train_oh, epochs = 50, batch_size = 32, shuffle=True)

X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
Y_test_oh = convert_to_one_hot(Y_test, C = 5)
loss, acc = model.evaluate(X_test_indices, Y_test_oh)
print()
print("Test accuracy = ", acc)

# check mislabeled examples
C = 5
y_test_oh = np.eye(C)[Y_test.reshape(-1)]
X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
pred = model.predict(X_test_indices)
for i in range(len(X_test)):
    x = X_test_indices
    num = np.argmax(pred[i])
    if(num != Y_test[i]):
        print('Expected emoji:'+ label_to_emoji(Y_test[i]) + ' prediction: '+ X_test[i] + label_to_emoji(num).strip())