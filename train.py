import keras
import json
import numpy as np
from sklearn.model_selection import train_test_split

#some constants
DATA_PATH = "/home/concina/machine_learning_python_book/keyword_spotting/data.json"
LEARNING_RATE = 0.0001
EPOCHS = 40
BATCH_SIZE = 32
SAVED_MODEL_PATH = "model.h5"

def load_dataset(data_path):
    with open(data_path, 'r') as fp:
        data = json.load(data_path)
    #extract input and target
    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    return X,y
    
def get_data_splits(data_path):
    #load the dataset
    X, y = load_dataset(data_path)
    #create train validation test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.1)
    
    #convert input from 2d to 3d array
    X_train = X_train[..., np.newaxis]
    X_validation = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return  X_train, X_validation, X_test,  y_train, y_validation, y_test
    
def build_model()
    pass
    
def main():
    
    #load train validation test data splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = get_data_splits(DATA_PATH)
    
    #build the CNN model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]) # num of segments, nm of coefficients 13, 1  (for a grey scale image is 1, RGB 3 , here is one since we only have mfcc)
    model = build_model(input_shape, LEARNING_RATE)
    
    #train the model
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_validation, y_validation))
    
    # evaluate the model using the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test error: {test_error}, test accuracy: {test_accuracy}")
    
    #save the model
    model.save(SAVED_MODEL_PATH)