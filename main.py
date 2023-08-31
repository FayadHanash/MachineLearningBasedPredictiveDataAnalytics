# Author: Fayad Al Hanash
# Date: 2023-08-30

# This file is the main file for the project
# import libraries
import numpy as np
from helper import Helper
import sklearn.model_selection as ms
from file_utility import FileUtility
from parameters_tunning import ParametersTunning
from binary_classification import BinaryClassification
from multi_classification import MultiClassification

# method that runs the binary classification for the given algorithm and dataset 
def binary_classification( alg, is_blanced, is_modeled, x_train, x_test, y_train, y_test):
    cl = BinaryClassification(alg,is_blanced, x_train, x_test, y_train, y_test)
    if is_modeled:
       mod = FileUtility.load_model(f'{alg}_bin_{is_blanced}_balancing')
       cl.set_model(mod)
    else:
        cl.fit()
        FileUtility.save_model(cl.get_model(), f'{alg}_bin_{is_blanced}_balancing')
    cl.display_matrix(f"{alg} bin for {is_blanced} dataset")
    cl.plot()


# method that runs the multi classification for the given algorithm and dataset
def multi_classification( alg, is_blanced, is_modeled, x_train, x_test, y_train, y_test):
    cl = MultiClassification(alg,is_blanced, x_train, x_test, y_train, y_test)
    if is_modeled:
       mod = FileUtility.load_model(f'{alg}_mul_{is_blanced}_balancing')
       cl.set_model(mod)
    else:
        cl.fit()
        FileUtility.save_model(cl.get_model(), f'{alg}_mul_{is_blanced}_balancing')
    cl.display_confusion_matrix(f"{alg} multi for {is_blanced} dataset")
    cl.plot()


# method that runs the parameters tunning for the given algorithm and dataset
def tune_parameters(alg,is_balanced,cl_type, x_train, y_train):
    pt = ParametersTunning(alg, x_train, y_train)
    para = pt.tune_parameters(is_balanced)
    FileUtility.save_text_json(para, f'{alg}_{cl_type}_{is_balanced}_balancing.txt')

# method that runs the print the cross validation for a already saved cross validation
def print_cross_validation(alg,is_balanced,cl_type):
    cr = FileUtility.load_model(f'{alg}_{cl_type}_{is_balanced}_balancing_cross')
    score = np.mean(cr['test_score'])
    print(f'Mean Accuracy: {score:.2%}')


# methid that runs the main method
def main():
 
    # read the data from the file
    xData, yData = FileUtility.read_file('data/extracted.csv')
    # split the data into train and test
    x_train, x_test, y_train, y_test = ms.train_test_split(xData, yData, train_size=0.80, test_size=0.20, random_state=0)
    # scale the data
    x_train, x_test = Helper.scale_min_max(x_train, x_test)
    
    
    algorithm = "svm"
    is_blanced = False
    cl_type = 'bin'
    is_modeled = True

    if cl_type == 'bin':
        binary_classification(algorithm, is_blanced, is_modeled, x_train, x_test, y_train, y_test)
    else:
        multi_classification(algorithm, is_blanced, is_modeled, x_train, x_test, y_train, y_test)  


# run the main method
if __name__ == "__main__":
    main()  