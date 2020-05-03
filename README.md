# LSTM-Program-Synthesis
Long Short Term Memory Recurrent Neural Network program synthesis
Initally, create a python3 virtual environment, activate it and run:
`pip3 install -r requirements.txt` 

# createTrainingData.py
This file is used to create training data. The location of the saved file and the amount of training data can be altered via command line arguments. Run `createTrainingData.py --help` To view the parameters

# train.py
This file is used to train a model on input data. Many parameters can be altered at run time. The defualt parameters will create the optimal model that was found
`python3 train.py --help` will list all paramters that can be changed. 
For instance, to train a model over 15 epohcs and call this model 'modelName':
`python3 train.py --model_name ModelName --epochs 15`
The defualt location for the model to be saved to is the models/ directory
# generateCode.py
This file takes a saved model and saves it's output as a python file.
The name of the model must be given at runtime:
`python3 generateCode.py --model ModelName.py`

This will save the output to a given file location. The default location will be data/output.
