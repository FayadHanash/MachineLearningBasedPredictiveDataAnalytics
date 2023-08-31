# The class Plot which is used to plot the data
# import the libraries
import numpy as np
import pandas as pd
import seaborn as sns
from helper import Helper
from matplotlib import pyplot as plt

class Plot:
    
    # method that plots the confusion matrix   
    @staticmethod 
    def plot_confusion_matrix_binary(cm,title):
        cm = cm
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    # method that plots the roc curve    
    @staticmethod
    def plot_roc_curve(fpr, tpr, roc_auc_score, title):
        fpr = fpr
        tpr = tpr
        plt.plot(fpr, tpr, label=f'ROC Curve: {roc_auc_score:.3}')
        plt.plot([0,1],[0,1], 'k--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.title(title)
        plt.show()

    # method that plots the confusion matrix for multi-class classification
    @staticmethod
    def plot_confusion_matrix_multi_class(y_pred, y_test, title):
        fig, ax = plt.subplots(figsize=(7,5))
        from sklearn.metrics import ConfusionMatrixDisplay
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues')
        ax.xaxis.set_ticklabels(Helper.get_the_classes())
        ax.yaxis.set_ticklabels(Helper.get_the_classes())
        _=ax.set_title(title)
        plt.show()

    # method that plots the correlation matrix 
    @staticmethod
    def plot_correlation_matrix(data, title):
        corr = np.corrcoef(data, rowvar=False)
        sns.heatmap(corr, annot=True, cmap='Blues')
        plt.title(title)
        plt.show()

    # method that plots the distribution of the classes both binary and multi-class classification
    @staticmethod
    def plot_class_distribution(x,str):
        class_count = pd.DataFrame(x).value_counts(sort = False)
        class_count.plot(kind = 'bar', rot=0)
        plt.title(f"{str}")
        plt.xlabel("Classes")
        plt.ylabel("Frequency")
        if len(class_count) <= 2 :
            plt.xticks(range(2), Helper.get_the_binary_classes())
        else:
            plt.xticks(range(4), Helper.get_the_classes())
        plt.show()


    # method that plots the histogram of each feature
    @staticmethod
    def plot_one_feature_histogram(data, title, xlabel, ylabel, bins=10):
        plt.barh(data,np.arange(len(data)))
        #plt.hist(data, bins=bins)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    # method that plots the histogram of all features
    @staticmethod
    def plot_features_histogram(data):
        num_features = data.shape[1]
        num_rows = (num_features +1) // 2
        fig, axes = plt.subplots(num_rows, 2, figsize=(10, 6 * num_rows))
        for i in range(num_features):
            ax = axes[i // 2, i % 2]
            ax.hist(data.iloc[:, i], bins=25)
            #ax.set_title(f"feature {i}")
            #ax.title.set_position([.5, 1.05])
            #ax.set_xlabel(xlabel)
            #ax.set_ylabel(ylabel)
        for i in range(num_features, num_rows * 2):
            fig.delaxes(axes.flatten()[i])
        plt.show()




            