#!/usr/bin/env python
from __future__ import print_function

import io
import sys
from time import time
from datetime import datetime
import random

import numpy as np

from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model
from keras.optimizers import RMSprop, SGD, Adagrad
from keras.utils.data_utils import get_file
from keras.callbacks import TensorBoard
#import matplotlib.pyplot as plt
import argparse

parse = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parse.add_argument('--file_path', type=str, default='data/trainFiles/input.txt',
                    help='file path containing input.txt')

parse.add_argument('--save_directory', type=str, default='models',
                    help='directory to store models')

parse.add_argument('--log_directory', type=str, default='logs/{}',
                    help='directory to store tensorboard logs')

parse.add_argument('--cells', type=int, default=128,
                    help='amount of cells in the RNN')

parse.add_argument('--sequence_length', type=int, default=40,
                    help='RNN sequence length. number of characters to train on.')

parse.add_argument('--step_size', type=int, default=3,
                    help='step size to jump through input.')

parse.add_argument('--batch_size', type=int, default=128,
                    help="batch size.")

parse.add_argument('--dropout_rate', type=int, default=0.2,
                    help='Dropout value after each layer.')

parse.add_argument('--epochs', type=int, default=12,
                    help='number of epochs.')

parse.add_argument('--learning_rate', type=float, default=0.01,
                    help='learning rate')

parse.add_argument('--decay_rate', type=float, default=0.00001,
                    help='decay rate')

parse.add_argument('--optimizer', type=str, default="adagrad",
                    help='Choose optimizer: rms, adagrad or sgd')

current_date = str(datetime.now()).split(' ')[0]
parse.add_argument('--model_name', type=str, default=current_date,
                    help='Set the name of the model (default is the current date)')

args = parse.parse_args()


#Function to read the input data
def read_file(file_path):
    try:
        with io.open(file_path, encoding='utf-8') as file:
            input_data = file.read().lower()
        return input_data
    except OSError:
        print("Could not open/read file: " + path)
        sys.exit()

# plot performance over the training epochs
# Unused as TensorBoard preferred, kept for historical reasons
# def show_results(nn_model_train):
#     for key in nn_model_train.history.keys():
#         print(key)
#     accuracy     = nn_model_train.history['accuracy']
#     loss         = nn_model_train.history['loss']
#     epochs       = range(len(accuracy))
#     nb_epochs    = len(epochs)
#     f2 = plt.figure(2)
#     plt.subplot(1,2,1)
#     plt.axis((0,nb_epochs,0,1.2))
#     plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
#
#     plt.title('Training accuracy')
#     plt.legend()
#     plt.subplot(1,2,2)
#     plt.axis((0,nb_epochs,0,1.2))
#     plt.plot(epochs, loss, 'bo', label='Training loss')
#     plt.title('Training loss')
#     plt.legend()
#     plt.draw()
#     plt.savefig('test2.png')
#     plt.pause(0.001)

#Split the input data into strings of length sequence length
def split_input_data(args, input_data):
    strings = []
    next_character = []
    for i in range(0, len(input_data) - args.sequence_length, args.step_size):
        strings.append(input_data[i: i + args.sequence_length])
        next_character.append(input_data[i + args.sequence_length])
    return strings, next_character

#Function to vectorise the strings and next character arrays
def vectorise(strings, sequence_length, distinct_characters, character_index,
              next_character):
    x_vector = np.zeros((len(strings), sequence_length,
                         len(distinct_characters)), dtype=np.bool)
    y_vector = np.zeros((len(strings), len(distinct_characters)), dtype=np.bool)

    #For each string in the strings[] array
    for i, string in enumerate(strings):
        #For each character in the current string
        for j, char in enumerate(string):
            #x_vector set to true for each string and character found
            x_vector[i, j, character_index[char]] = 1
        #y_vector is one step ahead of x_vector, it holds the next char
        y_vector[i, character_index[next_character[i]]] = 1
    return x_vector, y_vector

#Function to build and trian the LSTM neural network
def build_model(x, y, args, distinct_characters, tensorboard):

    model = Sequential()
    model.add(LSTM(args.cells
                   , input_shape=(args.sequence_length
                                  , len(distinct_characters))
                   , return_sequences=True))
    model.add(Dropout(args.dropout_rate))
    model.add(LSTM(args.cells))
    model.add(Dropout(args.dropout_rate))
    model.add(Dense(len(distinct_characters), activation='softmax'))
    if args.optimizer == 'adagrad':
        print("adagrad")
        optimizer = Adagrad(learning_rate=args.learning_rate, decay=args.decay_rate)
    elif args.optimizer == 'rms':
        print("rms")
        optimizer = RMSprop(learning_rate=args.learning_rate, decay=args.decay_rate)
    elif args.optimizer == 'sgd':
        optimizer = SGD(learning_rate=args.learning_rate, decay=args.decay_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    hist = model.fit(x
              , y
              , batch_size=args.batch_size
              , epochs=args.epochs
              , callbacks=[tensorboard])
    #show_results(hist)
    model.save(args.save_directory + '/' + args.model_name + '.h5')
    return hist

#Assing the input data to a varuable
input_data = read_file(args.file_path)

#Get a list of the distinct characters (vocabulary)
distinct_characters = sorted(list(set(input_data)))

print('total distinct characters:', len(distinct_characters))

#create a dictionary containing distinct charcaters and their index value
character_index = dict((character, index)
                        for index, character in enumerate(distinct_characters))

#Vaiables from this file are required in the generateCode.py file
#Therefore, this line of code is required so that when the generateCode.py file
#calls this file, the code within this if statement does not run
if __name__ == "__main__":
    # cut the input_data in semi-redundant sequences of sequence_length characters
    strings, next_character = split_input_data(args, input_data)

    #assign x (train) and y (validation) vectors
    x, y = vectorise(strings, args.sequence_length, distinct_characters,
                     character_index, next_character)
    #Tensorboard decleration
    tensorboard = TensorBoard(log_dir=str(args.log_directory).format(time()))
    #Send the command line arguments train and validation vectors and the
    #tensorboard object to the build model function
    build_model(x, y, args, distinct_characters, tensorboard)
