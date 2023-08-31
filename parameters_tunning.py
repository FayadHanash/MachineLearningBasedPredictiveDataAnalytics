# import libraries
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# The class ParametersTunning which is used to tune the parameters of the model
class ParametersTunning:
    # method that initializes the parameters tunning
    def __init__(self, alg_name, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.algorithm = None
        self.parameters = {}
        self.set_algorithm(alg_name)

    # method that initializes the algorithm and the parameters
    def set_algorithm(self, alg):
        if alg == "mlp":
            self.algorithm = MLPClassifier()
            self.parameters = {'hidden_layer_sizes': [(50,), (100,),(200,),(50,50)], 'activation': ['relu','tanh','sigmoid'], 'learning_rate': ['constant','adaptive'], 'solver': [ 'sgd', 'adam'],'max_iter':[200,300,400,500]}
        elif alg == "gb":
            self.algorithm = GradientBoostingClassifier()
            self.parameters = { 'n_estimators':[100,200,300],'criterion' : ['friedman_mse','squared_error'], 'max_features': [ 'sqrt', 'log2','auto'], 'loss': [ 'log_loss', 'deviance','exponential']}
        elif alg == "svm":
            self.algorithm = svm.SVC()
            self.parameters = { 'C': [0.1,1,1.5,2,2.5,3,4,4.5,5], 'kernel': [ 'rbf','poly'], 'degree' : [3,4],'gamma': ['scale', 'auto']}
        elif alg == "rf":
            self.algorithm = RandomForestClassifier()
            self.parameters = { 'n_estimators':[100,200,300],'class_weight' : ['balanced_subsample','balanced'], 'max_features': [ 'sqrt', 'log2','None'], 'criterion' : ['gini','entropy','log_loss']}
        else:
            raise ValueError("Invalid algorithm")
    
    # method that tunes the parameters of the model and returns the best parameters
    def tune_parameters(self, is_balanced):
        if is_balanced == False:
            return self.tune_parameters_unbalanced()
        else:
            return self.tune_parameters_balanced()

    # method that tunes the parameters of the model with unbalanced data and returns the best parameters
    def tune_parameters_unbalanced(self):
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
        #grid_search = GridSearchCV(estimator=self.algorithm, param_grid=self.parameters, cv=cv)
        grid_search = GridSearchCV(estimator=self.algorithm, param_grid=self.parameters)
        grid_search.fit(self.x_train, self.y_train)
        print(grid_search.best_params_)
        return grid_search.best_params_
    

    # method that tunes the parameters of the model with balanced data and returns the best parameters
    def tune_parameters_balanced(self):
        oversample = SMOTE()
        undersample = RandomUnderSampler()
        steps = [('o', oversample), ('u', undersample)]
        pipeline = Pipeline(steps=steps)
        x_train_res, y_train_res = pipeline.fit_resample(self.x_train, self.y_train)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
        grid_search = GridSearchCV(estimator=self.algorithm, param_grid=self.parameters, cv=cv)
        grid_search.fit(x_train_res, y_train_res)
        print(grid_search.best_params_)
        return grid_search.best_params_
