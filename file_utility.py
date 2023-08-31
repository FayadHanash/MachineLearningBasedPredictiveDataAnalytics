# FileUtility class to read and write files
import json
import pickle


class FileUtility:

    # method that reads file and returns x_data and y_data
    @staticmethod
    def read_file(filename):
        try:    
            x_data = []
            y_data = []
            with open(filename, 'r') as f:
                lines = f.readlines()
                for line in lines[1:lines.__len__()]:
                    line = line.strip().split(',')
                    x_data.append([float(line[0]), float(line[1]), float(line[2]),float(line[3]),float(line[4]),float(line[5]),float(line[6]),float(line[7])])
                    y_data.append(int(line[8]))
            return x_data, y_data
        except IOError as e:
            print("read method error", e.strerror)
    
    # method that writes x_data and y_data to file
    @staticmethod
    def write_to_file(x_train, x_test, filename):
        try:
            with open(filename, 'w') as f:
                for i in range(len(x_train)):
                    for j in range(len(x_train[i])):
                        f.write(str(x_train[i][j]))
                        f.write(',')
                    #f.write(str(y_train[i]))
                    f.write('\n')
                for i in range(len(x_test)):
                    for j in range(len(x_test[i])):
                        f.write(str(x_test[i][j]))
                        f.write(',')
                    #f.write(str(y_test[i]))
                    f.write('\n')
        except IOError as e:
            print("write method error", e.strerror)


    # method that saves the model            
    @staticmethod
    def save_model(model, filename):
        try: 
            with open(f"modeled/{filename}", 'wb') as f:
               pickle.dump(model, f)
        except IOError as e:
            print("save error", e.strerror)

    # method that loads the model
    @staticmethod
    def load_model(filename):
        try: 
            with open(f"modeled/{filename}", 'rb') as f:
                return pickle.load(f)
        except IOError as e:
            print("load error", e.strerror)
    
    # method that saves texts as json format
    @staticmethod
    def save_text_json(data, filename):
        try: 
            with open(f"parameters/{filename}", 'w') as f:
               json.dump(data,f)
        except IOError as e:
            print("save json error", e.strerror)