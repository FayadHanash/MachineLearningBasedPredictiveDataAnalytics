
# Importing the libraries
import numpy as np
from model import Model
from helper import Helper
from plot import Plot
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import multilabel_confusion_matrix, classification_report

# The class MultiClassification which is used to build the multi classification model
# inherits the Model class
class MultiClassification(Model):

    # method that initializes the multi classification model
    # calls the super class constructor
    def __init__(self, alg,is_blanced, x_train, x_test, y_train, y_test):
        super().__init__(x_train, x_test, y_train, y_test)
        self.is_blanced = is_blanced
        self.set_algorithm(alg)
        self.num_classes = len(np.unique(y_test))
    
    # method that initializes the algorithm
    def set_algorithm(self, alg):
        if alg == "mlp":
            if self.is_blanced == False:
                self.algorithm = MLPClassifier(activation='relu',hidden_layer_sizes=(50,50),learning_rate='adaptive',solver='adam',max_iter=200)
            else:
                self.algorithm = MLPClassifier(activation='relu',hidden_layer_sizes=(200,),learning_rate='adaptive',solver='adam')
        elif alg == "gb":
            if self.is_blanced == False:
                self.algorithm = GradientBoostingClassifier(n_estimators=300,criterion='squared_error',loss='log_loss',max_features='sqrt')
            else:
                self.algorithm = GradientBoostingClassifier(n_estimators=300,criterion='friedman_mse',loss='deviance',max_features='sqrt') 
        elif alg == "svm":
            if self.is_blanced == False:
                self.algorithm = svm.SVC(C=5,degree=4,gamma='scale',kernel='poly' , probability=True)
            else:
                self.algorithm = svm.SVC(C=3.5,degree=3, gamma='scale',kernel='poly', probability=True)
        elif alg == "rf":
            if self.is_blanced == False:
                self.algorithm = RandomForestClassifier(n_estimators=200,criterion='log_loss',max_features='sqrt')
            else:
                self.algorithm = RandomForestClassifier(n_estimators=300,criterion='log_loss',max_features='sqrt')
        else:
            raise ValueError("Invalid algorithm")
        
    # method that fits the model
    def fit(self):
        if self.is_blanced == False:
            self.fit_unbalanced_data()
        else:
            self.fit_balanced_data()
    
    # method that returns the multilabel confusion matrix
    def get_multilabel_confusion_matrix(self):
        return multilabel_confusion_matrix(self.y_test, self.y_predict)
    
    def get_classifcation_report(self):
        return classification_report(self.y_test, self.y_predict, target_names=["passed","failed","error","terminated"])
    
    # method that returns the accuracy, precision, recall and f1 score
    def get_accuracy_recall_precision_f1(self,arr):
        tn = arr[0][0]
        fp = arr[0][1]
        fn = arr[1][0]
        tp = arr[1][1]

        precision = 0
        recall = 0
        accuracy = 0
        f1 = 0
        # ignore the divide by zero warning
        np.seterr(divide='ignore', invalid='ignore') 
        
        if(tp+tn+fp+fn == 0):   
            accuracy = 0
            #raise ZeroDivisionError("Accuracy is undefined when TP+TN+FP+FN=0")
        else:
            accuracy = (tp+tn)/(tp+tn+fp+fn)
        if(tp == 0):
            precision = 0
        elif (tp+fp == 0):
            precision = 0
            #raise ZeroDivisionError("Precision is undefined when TP+FP=0")
        #elif (fp == 0):
         #   precision = "precision is not estimated when FP=0, the value is 1"
        else:
            precision = tp/(tp+fp)
        if (tp + fn == 0):
            recall = 0
            #raise ZeroDivisionError("Recall is undefined when TP+FN=0")
        #elif (fn == 0):
         #   recall = "recall is not estimated when FN=0, the value is 1"
        else:
            recall = tp/(tp+fn)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return accuracy,precision,recall,f1
    
    # method that returns the kappa and mcc
    def get_kappa_mcc(self, arr):
        tn = arr[0][0]
        fp = arr[0][1]
        fn = arr[1][0]
        tp = arr[1][1]
        
        # ignore the divide by zero warning
        np.seterr(divide='ignore', invalid='ignore')

        kappa_divisor = (tp+fp)*(fp+tn)+(tp+fn)*(fn+tn)
        if (kappa_divisor == 0):
            kappa = 0
            #raise ZeroDivisionError("Kappa is undefined when divisor=0")
        else:
            kappa = (tp+tn)/kappa_divisor
        mcc_divisor = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        if (mcc_divisor == 0):
            mcc = 0
            #raise ZeroDivisionError("MCC is undefined when divisor=0")
        else:
            mcc = (tp*tn-fp*fn)/mcc_divisor
        return kappa,mcc

    # method that displays the confusion matrix
    def display_confusion_matrix(self,title):
        print("*********************************************************")
        cm = self.get_multilabel_confusion_matrix()
        print(f"Confusion Matrix for {title}:\n",cm)
        for i in range(self.num_classes):
            print(f"Class: ({Helper.get_status(i)})")
            accuracy, precision, recall, f1 = self.get_accuracy_recall_precision_f1(cm[i])
            kappa, mcc = self.get_kappa_mcc(cm[i])
            print(f"Accuracy:\t{accuracy:.2%}\nPrecision:\t{precision:.3}\nRecall:\t\t{recall:.3}\nF1 Score:\t{f1:.3}\nKappa:\t\t{kappa:.3}\nMCC:\t\t{mcc:.3}")
            #(f"Cohen's Kappa Score:\t{kappa:.3}\nMatthews Correlation Coefficient:\t{mcc:.3}")
        print("*********************************************************")

    # method that plots the confusion matrix
    def plot(self):
        Plot.plot_confusion_matrix_multi_class(self.y_predict,self. y_test, f"Confusion Matrix multi-class {self.algorithm.__class__.__name__}")

    # method that displays the classification report
    def display_report(self):
        print("*********************************************************")
        print("Classification Report:\n",self.get_classifcation_report())
        print("*********************************************************")








