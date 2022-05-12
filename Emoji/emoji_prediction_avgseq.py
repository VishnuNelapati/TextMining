'''
method: use the average embedding vectors of a sentence in prediction

predict emoji given a sentence, such as 
never talk to me again 😞
I am proud of your achievements 😄
It is the worst day in my life 😞
Miss you so much ❤️
food is life 🍴
I love you mum ❤️
Stop saying bullshit 😞
congratulations on your acceptance 😄
The assignment is too long  😞
I want to go play ⚾

'''

import os
os.chdir(r'C:\all\Desktop\Course Slides\TextMining\2022spring\11 text classification with deep learning\sentiment_analysis_with_deeplearning')

import numpy as np
import emoji
import matplotlib.pyplot as plt
from keras.layers import Dense,Input
from keras.models import Model
from tensorflow.keras.optimizers import Adam
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
    return emoji.emojize(emoji_dictionary[str(label)], use_aliases=True)

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

# step2: load and preprocess the train/test dataset
# X_train: (m,) shape array, each value is a string, m is the number of training examples
# y_train: (m,) shape array, each value is a emoji index.
X_train, Y_train = read_csv('train_emoji.csv')
X_test, Y_test = read_csv('test_emoji.csv')

# check the first 10 doc
for idx in range(10):
    print(X_train[idx], label_to_emoji(Y_train[idx]))
    
# convert to one hot
# Y_oh_train is a (m,5) shape, 5 is the number of different emoji index
Y_oh_train = convert_to_one_hot(Y_train, C = 5)
Y_oh_test = convert_to_one_hot(Y_test, C = 5)

# load glove word embedding
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')


# step3: prepare train/test for deep learning
# convert sentence to word embedding presentation
def sentence_to_avg(sentence, word_to_vec_map):
    """
    first converts a sentence (string) into a list of words (strings). Extracts the GloVe representation of each word
    and averages their values into a single vector encoding the meaning of the sentence.
    
    Arguments:
    sentence -- string, one training example from X
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    
    Returns:
    avg -- average vector encoding information about the sentence, numpy-array of shape (50,)
    """
    
    # Step 1: Split sentence into list of lower case words (≈ 1 line)
    words = sentence.lower().split()

    # Initialize the average word vector, should have the same shape as your word vectors.
    avg = np.zeros((50,))
    
    # Step 2: average the word vectors. You can loop over the words in the list "words".
    total = 0
    for w in words:
        total += word_to_vec_map[w]
    avg = total/len(words)
    
    return avg

avg = sentence_to_avg("what is your favorite baseball game", word_to_vec_map)
print("avg = \n", avg)

# convert X_train to X_train_step2 (same shape but change each sentence to a (50,) np array)
X_train_step1=[list(sentence_to_avg(sentence, word_to_vec_map)) for sentence in X_train]
X_train_step2=np.array(X_train_step1) 
X_test_step1=[list(sentence_to_avg(sentence, word_to_vec_map)) for sentence in X_test]
X_test_step2=np.array(X_test_step1) 

# step4: deep learning
def model(embed_length):
    '''
    define the model, in the model definition, we only show calculation for one training example
    
    '''
    X=Input(shape=(embed_length)) # the final result will be (m,embed_length) input
    densor1=Dense(20,activation="tanh")
    densor2=Dense(10,activation="tanh")
    densor3=Dense(5,activation="softmax") # there are 5 classes
    output1=densor1(X)
    output2=densor2(output1) 
    output3=densor3(output2) # the current output 3 is a (5,) array, but the final result in training will be a (m,5) array
    
    model=Model(inputs=X,outputs=output3)
    return model

model = model(50)
model.summary()
opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_step2, Y_oh_train, epochs=100, batch_size=10) # the Y_oh_train is already shape (m,5), so it matches the output of the defined model.

predicted_prob=model.predict(X_test_step2) # will be a (56,5) shape, 56 is the number of testing examples. Each row is the predicted prob of each of the five emoji.

for index, value in enumerate(list(predicted_prob)):
    print(X_test[index],label_to_emoji(np.argmax(value)),label_to_emoji(str(Y_test[index]))) # text, predicted label, true label




    
    
    
    







