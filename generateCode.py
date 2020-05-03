import random
import os
import sys

import json
import numpy as np
import argparse
from keras.models import load_model

import train



args = train.args

parse = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Data and model checkpoints directories
parse.add_argument('--model', type=str,
                    help='The name of the model to be loaded')
parse.add_argument('--chars_to_print', type=int, default=650,
                    help='The amount of characters to output')

args = parse.parse_args()

#Reverse of the character index dictionary
#This dictionary contains the index: character tuple
index_character = dict((index, character)
                  for index, character in enumerate(train.distinct_characters))

# function to get the next character index from probability array
def get_next_index(prediction, temperature):
    # Convert the prediction array to float 64 values
    prediction = np.asarray(prediction).astype('float64')
    with np.errstate(divide='ignore'):
        #divide the natural logarithm of each value by the temperature
        prediction = np.log(prediction) / temperature
    #reverse the logarithm
    exp_prediction = np.exp(prediction)
    #divide each element in the prediction array by the sum of the prediction array
    prediction = exp_prediction / np.sum(exp_prediction)
    #1 experiment, get the probability of each index
    probability = np.random.multinomial(1, prediction, 1)
    #The max value of the probability array is retured
    next_index = np.argmax(probability)

    return next_index




def output_shape_code(model, seq_length):
    #Get a random start index from the input data
    start_index = random.randint(0
                                 , len(train.input_data)
                                   - seq_length
                                   - 1)
    #Initalise output string
    output_string = ""
    #Tempearture is set to 1.2
    #Temperature ensures randomness in predictions
    temperature = 1.2

    #Set the randomised seed string
    current_string = train.input_data[start_index:
                                      start_index + seq_length]#args.sequence_length]
    #current_string = "import turtle\n"
    #current_string += "import autopy\n"

    print("current string = "+current_string)

    #each iteartion appends a character to the ouput string

    for i in range(650):
        #Declare new Predition vector
        x_prediction = np.zeros((1
                           , seq_length
                           , len(train.distinct_characters)))
        #Populate prediction vector
        for j, character in enumerate(current_string):
            x_prediction[0, j, train.character_index[character]] = 1.
        #Call the predict method on the model
        prediction = model.predict(x_prediction, verbose=0)[0]
        #Pass prediction vector and temperature to the get_next_index function
        #Next index is returned by the function and assined to variable
        next_index = get_next_index(prediction, temperature)
        #Get the predicted character by passing the next_index variable
        #to the index character dictionary
        next_character = index_character[next_index]
        #Add the predicted character to the end of the seed string and remove
        #the first character to keep the seed string of a constant sequence
        #length
        current_string = current_string[1:] + next_character

        #Append the predicted character to the output string
        output_string += next_character
    #Return the output string
    return output_string

#Load the saved model
#os.environ["PATH"] += os.pathsep + '/home/calkey/.local/lib/python3.5/site-packages'
#model_name = ''
model_name = args.model
#model_name = "models/z.h5"
model = load_model('models/'+model_name)
print(model.to_json())
#from keras.utils import plot_model
#plot_model(model, to_file=model_name+'.png', show_shapes=True, expand_nested=True)

#Pass model to the output_shape_code function
output_string = output_shape_code(model, 40)
print(output_string)
#remove any erroneous lines above the first 'import' in the output string
output_string = output_string[output_string.find('import'):]
#trim the string on 16 new line characters
output_string = os.linesep.join(output_string.split(os.linesep)[:16])
#Output the final string
print("output: \n"+output_string)
save_output_path = "data/output/output"+str(random.randint(1,9999))+".py"
with open(save_output_path, "w") as out_file:
    #write the output string to the file
    out_file.write(output_string)
    print("Output written to file: "+save_output_path)
