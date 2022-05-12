import os
os.chdir(r'C:\all\Desktop\Course Slides\TextMining\2021fall\08 Sentiment analysis\sentiment_analysis_with_deeplearning')

import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM
from emo_utils import *
import emoji

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

def sentences_to_embedding(X, word_to_vec_map, max_len, embedding_length=50):
    """
    Converts an array of sentences (strings) into an array of word embedding vectors corresponding to words in the sentences.
    
    args:
        X: array of sentences (strings), of shape (m, 1)
        word_to_vec_map: a dictionary containing the each word mapped to its embedding vector
        max_len: maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
        embedding_length: 50 in this case when using glove.6B.50d.txt
    
    return:
    data_array: array of embedding vectors corresponding to words in the sentences from X, of shape (m, max_len, embedding_length)
    """
    
    m = X.shape[0] # number of training examples
    
    # Initialize data_array as a numpy matrix of zeros and the correct shape
    data_array = np.zeros((m, max_len, embedding_length))
    
    for i in range(m): # loop over training examples
        
        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words =X[i].lower().split()
        
        # Loop over the words of sentence_words
        for index_w,w in enumerate(sentence_words):
            data_array[i, index_w,:] = word_to_vec_map.get(w,np.zeros((50))) # in this case, if the sentence has less than max_len words, we don't need to pad, since the data_array is initialized to have the zeros
        # also note that the max_len is the len of the longest sentence, so we will not get index out of bound error
    return data_array

X1 = np.array(["funny lol", "lets play baseball", "food is ready for you"])
X1_data_array = sentences_to_embedding(X1,word_to_vec_map, max_len=5)
print("X1 =", X1)
print("X1_indices =\n", X1_data_array)

X_train_data_array=sentences_to_embedding(X_train, word_to_vec_map, maxLen)
Y_train_oh = convert_to_one_hot(Y_train, C = 5)
X_test_data_array=sentences_to_embedding(X_test, word_to_vec_map, maxLen)
Y_test_oh = convert_to_one_hot(Y_test, C = 5)

# define and train deep learning model
def model(input_shape):
    """
    args:
        input_shape: shape of the input, (max_len, embedding_length)
    
    returns:
        model: a model instance in Keras
    """
    sentence = Input(shape=input_shape) # shape is (,max_len, embedding_length)
            
    # check the keras_lstm_instruction.py before moving on.
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden states
    # The returned output should be a batch of sequences.
    X = LSTM(units=128, return_sequences=True)(sentence) # shape is (,max_len,128), max_len is like the number of time steps, hidden state is the output of each block, cell state (memory state) is different. But they have the same units
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X) # The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting.  shape is still (,max_len,128)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # The returned output should be a single hidden state/output, not a batch of sequences.
    X = LSTM(units=128,return_sequences=False)(X) # shape is (,128), this is the second floor of the deep lstm
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X) # shape is still (,128)
    # Propagate hidden state through a Dense layer with 5 units
    y = Dense(5,activation="softmax")(X) # shape (,5)

    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence, outputs=y)
    
    return model

model = model((maxLen,50))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_data_array, Y_train_oh, epochs = 50, batch_size = 32)

loss, acc = model.evaluate(X_test_data_array, Y_test_oh)
print("Test accuracy = ", acc)

# check mislabeled examples
C = 5
pred = model.predict(X_test_data_array)
for i in range(X_test.shape[0]):
    num = np.argmax(pred[i])
    if(num != Y_test[i]):
        print('Expected emoji:'+ label_to_emoji(Y_test[i]) + ' prediction: '+ X_test[i] + label_to_emoji(num).strip())