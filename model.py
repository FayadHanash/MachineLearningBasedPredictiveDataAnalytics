# import libraries
from numpy import mean
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_validate ,cross_val_score, RepeatedStratifiedKFold

# The class Model which is used to build the model
class Model:
    # method that initializes the model
    def __init__(self, x_train, x_test, y_train, y_test):
        self.model = None
        self.algorithm = None
        self.score = None
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_predict = None
        self.y_predict_proba = None
        self.cross_validation = None
        
    # method that fits the model
    def fit(self):
        pass 

    # method that fits the model with unbalanced data
    def fit_unbalanced_data(self):
        self.model = self.algorithm.fit(self.x_train, self.y_train)
        self.score = self.model.score(self.x_test, self.y_test)
        self.y_predict = self.model.predict(self.x_test)
        self.y_predict_proba = self.model.predict_proba(self.x_test)
        
    # method that fits the model with balanced data
    def fit_balanced_data(self):
        try:
            oversample = SMOTE()
            undersample = RandomUnderSampler()
            steps = [('o', oversample), ('u', undersample)]
            pipeline = Pipeline(steps=steps)
            x_train_res, y_train_res = pipeline.fit_resample(self.x_train, self.y_train)
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
            self.cross_validation = cross_validate(self.algorithm, x_train_res, y_train_res, scoring='accuracy', cv=cv, return_estimator=True)
            self.score = mean(self.cross_validation['test_score'])
            print(f'Mean Accuracy: {self.score:.2%}')
            #self.model = cross_val['estimator'][0].fit(x_train_res, y_train_res)
            self.model = self.algorithm.fit(x_train_res, y_train_res)
            self.y_predict = self.model.predict(self.x_test)
            self.y_predict_proba = self.model.predict_proba(self.x_test)
        except Exception as ex:
            raise ex
        
    # method that predicts and returns the result
    def predict(self, x):
        return self.model.predict(x)
    
    # method that predicts and returns the probability
    def predict_proba(self, x):
        return self.model.predict_proba(x)
    
    # method that returns the model's score
    def get_score(self):
        return self.score
    
    # method that returns the model
    def get_model(self):
        return self.model
    
    # method that sets the model and predicts the result and probability for the test data
    def set_model(self,model):
        self.model = model
        self.y_predict = self.model.predict(self.x_test)
        self.y_predict_proba = self.model.predict_proba(self.x_test)

    # method that sets the cross validation
    def set_cross_validation(self,cross):
        self.cross_validation = cross  

    # method that returns the cross validation      
    def get_cross_validation(self):
        return self.cross_validation



