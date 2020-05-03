# LSTM-Program-Synthesis
Long Short Term Memory Recurrent Neural Network program synthesis

# createTrainingData.py
This file is used to create training data. There are two command line arguemnts that can be changed:

# train.py
This file is used to train a model on input data. Many parameters can be altered at run time. 
'python3 train.py --help' will list all paramters that can be changed. For instance, to name the saved model simply run train.py with the parameter:
'python3 train.py --model_name yourModelName'

# generateCode.py
This file takes a saved model and saves it's output as a python file.
The name of the model must be given at runtime:
'python3 generateCode.py --model yourModelName.py'

This will save the output to a given file location. The default location will be data/output.
