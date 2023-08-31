# Importing the libraries
from plot import Plot
from sklearn import svm
from model import Model
from helper import Helper
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,matthews_corrcoef,cohen_kappa_score, roc_curve

# The class BinaryClassification which is used to build the binary classification model
# inherits the Model class
class BinaryClassification(Model):
    # method that initializes the binary classification model
    # calls the super class constructor
    # calls the binarize method from the helper class to binarize the target features
    def __init__(self, alg, is_blanced, x_train, x_test, y_train, y_test):
        self.is_blanced = is_blanced
        super().__init__(x_train, x_test, Helper.binarize(y_train), Helper.binarize(y_test))
        self.set_algorithm(alg)

    # method that initializes the algorithm    
    def set_algorithm(self, alg):
        if alg == "mlp":
            if self.is_blanced == False:
                self.algorithm = MLPClassifier(activation='relu',hidden_layer_sizes=(200,),learning_rate='constant',solver='adam')
            else:
                self.algorithm = MLPClassifier(activation='relu',hidden_layer_sizes=(50,50),learning_rate='adaptive',solver='adam', max_iter=300)
        elif alg == "gb":
            if self.is_blanced == False:
                self.algorithm = GradientBoostingClassifier(n_estimators=300,criterion='squared_error',loss='deviance',max_features='sqrt')
            else:
                self.algorithm = GradientBoostingClassifier(n_estimators=300,criterion='friedman_mse',loss='log_loss',max_features='log2')
        elif alg == "svm":
                self.algorithm = svm.SVC( C=5,degree=3,gamma='scale',kernel='poly' ,probability=True)
        elif alg == "rf":
            if self.is_blanced == False:
                self.algorithm = RandomForestClassifier(n_estimators=100,criterion='log_loss',max_features='sqrt')
            else:
                self.algorithm = RandomForestClassifier(n_estimators=200,criterion='gini',max_features='sqrt')
        else:
            raise ValueError("Invalid algorithm")

    # method that fits the model    
    def fit(self):
        if self.is_blanced == False:
            self.fit_unbalanced_data()
        else:
            self.fit_balanced_data()
    
    # method that returns the confusion matrix
    def get_confusion_matrix(self):
        return confusion_matrix(self.y_test, self.y_predict)
    
    # method that returns the accuracy score
    def get_accuracy(self):
        return accuracy_score(self.y_test, self.y_predict)
    
    # method that returns the precision score
    def get_precision(self):
        return precision_score(self.y_test, self.y_predict)
    
    # method that returns the recall score
    def get_recall(self):
        return recall_score(self.y_test, self.y_predict)
    
    # method that returns the f1 score
    def get_f1_score(self):
        return f1_score(self.y_test, self.y_predict)
    
    # method that returns the roc auc score
    def get_roc_auc_score(self):
        return roc_auc_score(self.y_test, self.model.predict_proba(self.x_test)[:,1])
    
    # method that returns the roc curve
    def get_roc_curve(self):
        return roc_curve(self.y_test, self.model.predict_proba(self.x_test)[:,1])
    
    # method that returns the matthews corrcoef score
    def get_matthews_corrcoef(self):
        return matthews_corrcoef(self.y_test, self.y_predict)
    
    # method that returns the cohens kappa score
    def get_cohen_kappa_score(self):
        return cohen_kappa_score(self.y_test, self.y_predict)
    
    # method that displays the measures
    def display_matrix(self, title):
        print("*********************************************************")
        print(f"Confusion Matrix: {title}\n\n{self.get_confusion_matrix()}\n")
        print(f"Accuracy:\t{(self.get_accuracy()):.2%}")
        print(f"Precision:\t{(self.get_precision()):.3}")
        print(f"Recall:\t\t{(self.get_recall()):.3}")
        print(f"F1 Score:\t{(self.get_f1_score()):.3}")
        print(f"Kappa:\t\t{(self.get_cohen_kappa_score()):.3}")
        print(f"ROC AUC:\t{(self.get_roc_auc_score()):.3}")
        print(f"MCC:\t\t{(self.get_matthews_corrcoef()):.3}")
        print("*********************************************************")

    # method that plots the confusion matrix and roc curve
    def plot(self):
        Plot.plot_confusion_matrix_binary(self.get_confusion_matrix(), f"Confusion Matrix binary {self.algorithm.__class__.__name__}")
        Plot.plot_roc_curve(self.get_roc_curve()[0], self.get_roc_curve()[1], self.get_roc_auc_score(), f"ROC Curve binary {self.algorithm.__class__.__name__}")
            