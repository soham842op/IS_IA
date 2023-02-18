import pandas as pd
import numpy as np
from pandas import ExcelWriter
from pandas import ExcelFile
import re
from sklearn.metrics import confusion_matrix 
# from sklearn.cross_validation import train_test_split 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.naive_bayes import GaussianNB 
from sklearn import metrics 
import matplotlib.pyplot as plt
#importing wx files
import wx
#import the newly created GUI file

from sklearn import svm
import webbrowser
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from yellowbrick.classifier import ClassificationReport
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,roc_curve
from sklearn import preprocessing



#extract single feature
def extract_feature_usertest(url):
    
    
    #length of url
    l_url=len(url)
    if(l_url > 54):
        length_of_url = 1
    else:
        length_of_url = 0

    
    #url has http
    if (("http://" in url) or ("https://" in url)):
        http_has = 1
    else:
        http_has = 0

    #url has suspicious char
    if (("@" in url) or ("//" in url)):
        suspicious_char = 1
    else:
        suspicious_char = 0


    #prefix or suffix
    if ("-" in url):
        prefix_suffix = 1
    else:
        prefix_suffix = 0

    #no of dots
    if ("." in url):
        count = len(url.split('.'))-1
        if (count > 5):
            dots = 0
        else:
            dots = 1
    else:
        dots = 0
    
    #no of slash
    if ("/" in url):
        count = len(url.split('/'))-1
        if (count > 5):
            slash = 0
        else:
            slash = 1
    else:
        slash = 0

    #url has phishing terms
    #("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)
    if (("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)):
        phis_term = 1
    else:
        phis_term = 0
    
    #length of subdomain
    it = url.index("//") + 2
    j = url.index(".")
    c = j - it;
    if (c > 5):
        sub_domain = 0
    else:
        sub_domain = 1
    
    #url contains ip address
    if re.match("\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b",url):
        ip_contain = 1
    else:
        ip_contain = 0
    
    
    return length_of_url,http_has,suspicious_char,prefix_suffix,dots,slash,phis_term,sub_domain,ip_contain







#extract testing feature
def extract_feature_test(url,output):
    
    
    #length of url
    l_url=len(url)
    if(l_url > 54):
        length_of_url = 1
    else:
        length_of_url = 0

    
    #url has http
    if (("http://" in url) or ("https://" in url)):
        http_has = 1
    else:
        http_has = 0

    #url has suspicious char
    if (("@" in url) or ("//" in url)):
        suspicious_char = 1
    else:
        suspicious_char = 0


    #prefix or suffix
    if ("-" in url):
        prefix_suffix = 1
    else:
        prefix_suffix = 0

    #no of dots
    if ("." in url):
        count = len(url.split('.'))-1
        if (count > 5):
            dots = 0
        else:
            dots = 1
    else:
        dots = 0
    
    #no of slash
    if ("/" in url):
        count = len(url.split('/'))-1
        if (count > 5):
            slash = 0
        else:
            slash = 1
    else:
        slash = 0

    #url has phishing terms
    #("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)
    if (("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)):
        phis_term = 1
    else:
        phis_term = 0
    
    #length of subdomain
    it = url.index("//") + 2
    j = url.index(".")
    c = j - it;
    if (c > 5):
        sub_domain = 0
    else:
        sub_domain = 1
    
    #url contains ip address
    if re.match("\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b",url):
        ip_contain = 1
    else:
        ip_contain = 0
    
    #output
    yn = output
    
    return yn,length_of_url,http_has,suspicious_char,prefix_suffix,dots,slash,phis_term,sub_domain,ip_contain

#extract training feature
def extract_feature_train(url,output):
    
    
    #length of url
    l_url=len(url)
    if(l_url > 54):
        length_of_url = 1
    else:
        length_of_url = 0

    
    #url has http
    if (("http://" in url) or ("https://" in url)):
        http_has = 1
    else:
        http_has = 0

    #url has suspicious char
    if (("@" in url) or ("//" in url)):
        suspicious_char = 1
    else:
        suspicious_char = 0


    #prefix or suffix
    if ("-" in url):
        prefix_suffix = 1
    else:
        prefix_suffix = 0

    #no of dots
    if ("." in url):
        count = len(url.split('.'))-1
        if (count > 5):
            dots = 0
        else:
            dots = 1
    else:
        dots = 0
    
    #no of slash
    if ("/" in url):
        count = len(url.split('/'))-1
        if (count > 5):
            slash = 0
        else:
            slash = 1
    else:
        slash = 0

    #url has phishing terms
    #("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)
    if (("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)):
        phis_term = 1
    else:
        phis_term = 0
    
    #length of subdomain
    it = url.index("//") + 2
    j = url.index(".")
    c = j - it;
    if (c > 5):
        sub_domain = 0
    else:
        sub_domain = 1
    
    #url contains ip address
    if re.match("\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b",url):
        ip_contain = 1
    else:
        ip_contain = 0
    
    #output
    yn = output

    return yn,length_of_url,http_has,suspicious_char,prefix_suffix,dots,slash,phis_term,sub_domain,ip_contain

#import train data
def importdata_train(): 
    balance_data = pd.read_csv('feature_train.csv',sep= ',', header = 1,usecols=range(1,11),encoding='utf-8') 
      
      
    # Printing the dataswet shape 
    print ("Dataset Lenght: ", len(balance_data)) 
    print ("Dataset Shape: ", balance_data.shape) 
      
    # Printing the dataset obseravtions 
    print ("Dataset: ",balance_data.head()) 
    return balance_data 
#import test data
def importdata_test(): 
    balance_data = pd.read_csv('feature_test.csv',sep= ',', header = 1,usecols=range(1,11),encoding='utf-8') 
      
      
    # Printing the dataswet shape 
    print ("Dataset Lenght: ", len(balance_data)) 
    print ("Dataset Shape: ", balance_data.shape) 
      
    # Printing the dataset obseravtions 
    print ("Dataset: ",balance_data.head()) 
    return balance_data 
#split data into train and test
def splitdataset(balance_data): 
  
    # Seperating the target variable 
    X = balance_data.values[:, 1:10]
    Y = balance_data.values[:, 0] 
  
    # Spliting the dataset into train and test 
    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100) 
      
    return X, Y
# Function to perform training with entropy. 
def tarin_using_entropy(X_train, y_train): 
  
    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 2, min_samples_leaf = 10) 
  
    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy 
# Function to make predictions 
def prediction(X_test, clf_object): 
  
    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test) 
    #print("Predicted values:") 
    #print(y_pred) 
    return y_pred 
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
      
    print("Report : ", 
    classification_report(y_test, y_pred))

    return accuracy_score(y_test,y_pred)*100

#roc
def plot_roc_curve(fpr, tpr ):  
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

