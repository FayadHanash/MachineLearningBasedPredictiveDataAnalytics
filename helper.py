# A helper class for the main program
class Helper:

    # method that binarizes the data
    @staticmethod
    def binarize(y_data):
        for i in range(len(y_data)):
            if y_data[i] == 0:
                y_data[i] = 0
            else:
                y_data[i] = 1
        return y_data
    

    # method that returns the classes of the target feature as a list
    @staticmethod
    def get_the_classes():
        return ['passed','failed','error','terminated']

    # method that returns the binary classes of the target feature as a list
    @staticmethod
    def get_the_binary_classes():
        return ['passed','rest']
    

    # method that normalizes the data using the min max scaler 
    # returns the normalized train and test data
    @staticmethod
    def scale_min_max(x_train, x_test):
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        s_train = scaler.fit_transform(x_train)
        s_test = scaler.transform(x_test)
        return s_train, s_test
    
    # method that returns the status of the target feature    
    @staticmethod
    def get_status(i):
        if i ==0: 
            x = "passed"
        elif i ==1:
            x = "failed"
        elif i ==2:
            x = "error"
        elif i ==3:
            x = "terminated"
        else:
            x = "other"
        return x