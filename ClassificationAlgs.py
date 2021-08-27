# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 23:49:48 2021

@author: alireza
"""
"""
Import packages
"""
import pandas as pd  # DataFram Import
import numpy as np   # Numpy Import

File_Addr_To_Save="E:\\Dataset\\140004_Dataset_NargesMohammadi\\Scenario A1\\ALLData.arff"

""" shuffle Samples """

from sklearn.utils import shuffle

DS_T=pd.read_csv(File_Addr_To_Save)
DS=shuffle(DS_T) ## We shuffle the dataset to improve randome distribution


""" 
Note: Before this code file, ImportMergeDataset code file should be executed to fill DS as a dataset 
"""



""" Set X & Y """

Samples_count=DS.shape[0]
Attributes_count=DS.shape[1]
print("The number of samples is:",Samples_count,"........","The number of Attributes is:" ,Attributes_count )



""" __________________________Set Train, Eval & Test Set ____________________________"""


per_train=0.6           # 60% of samples is trained set
per_eval=0.2            # 20% of samples is evaluated set
per_test=0.2            # 20% of samples is test set

""" Set Train Set """
train_set=DS[0:round(per_train*Samples_count)][:]
print(train_set.shape)
""" Set Evaluation Set """
eval_set=DS[round(per_train*Samples_count):round(per_train*Samples_count)+round(per_eval*Samples_count)][:]
print(eval_set.shape)
""" Set Test Dastaset """
test_set=DS[round(per_train*Samples_count)+round(per_eval*Samples_count):round(per_train*Samples_count)+round(per_eval*Samples_count)+round(per_test*Samples_count)][:]
print(test_set.shape)
""" ------------  """


""" _________________________________________________________________________________"""

""" _______________________________ Set X & Y _______________________________________"""
X_col=['duration', 'total_fiat', 'total_biat', 'min_fiat', 'min_biat','max_fiat', 'max_biat', 'mean_fiat', 'mean_biat', 'flowPktsPerSecond','flowBytesPerSecond', 'min_flowiat', 'max_flowiat', 'mean_flowiat','std_flowiat', 'min_active', 'mean_active', 'max_active', 'std_active','min_idle', 'mean_idle', 'max_idle', 'std_idle']



##  Train Set Regulation 
train_set_x=train_set[:][X_col]
train_set_y=train_set[:]['class1']

train_set_y[train_set_y=='Non-VPN']=0 ## Set Y if is Non-VPN = 0 
train_set_y[train_set_y=='VPN']=1     ## Set  if is VPN = 1


train_set_x=np.array(train_set_x)  ## X--> Each Column is a sampel 
                                     ## X--> Each Row is a feature
train_set_y=np.array(train_set_y)


## Eval_Set Regulation
eval_set_x=eval_set[:][X_col]
eval_set_y=eval_set[:]['class1']

eval_set_y[eval_set_y=='Non-VPN']=0 ## Set Y if is Non-VPN = 0 
eval_set_y[eval_set_y=='VPN']=1     ## Set  if is VPN = 1


eval_set_x=np.array(eval_set_x)  ## X--> Each Column is a sampel 
                                     ## X--> Each Row is a feature
eval_set_y=np.array(eval_set_y)


## Test Set Regulation
test_set_x=test_set[:][X_col]
test_set_y=test_set[:]['class1']

test_set_y[test_set_y=='Non-VPN']=0 ## Set Y if is Non-VPN = 0 
test_set_y[test_set_y=='VPN']=1     ## Set  if is VPN = 1


test_set_x=np.array(test_set_x)  ## X--> Each Column is a sampel 
                                     ## X--> Each Row is a feature
test_set_y=np.array(test_set_y)

""" Print Train, Eval and Test Set Shape """
print("x Train shape:",train_set_x.shape , "------------- ", "y shape:", train_set_y.shape)
print("x Eval shape:",eval_set_x.shape , "------------- ", "y shape:", eval_set_y.shape)
print("x Test shape:",test_set_x.shape , "------------- ", "y shape:", test_set_y.shape)
""" ------------------------------------"""


""" Set binary to ys """
train_set_y[train_set_y==0]=bin(0) ## Set binary 0
train_set_y[train_set_y==1]=bin(1) ## Set binary 1

eval_set_y[eval_set_y==0]=bin(0)  ## Set binary 0
eval_set_y[eval_set_y==1]=bin(1)  ## Set binary 1

test_set_y[test_set_y==0]=bin(0)  ## Set binary 0
test_set_y[test_set_y==1]=bin(1)  ## Set binary 1

""" ________________________________________________________________________________"""

""" Classification1: Logistic Regression """

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0,multi_class='ovr',C=10.0).fit(train_set_x,train_set_y) # 

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(train_set_y, clf.predict(train_set_x)),"<--Train Set Evaluation")
print(classification_report(eval_set_y, clf.predict(eval_set_x)),"<--Eval Set Evaluation")
print(classification_report(test_set_y, clf.predict(test_set_x)),"<--Test Set Evaluation")
"""____________________________________________________________"""

""" Classification2: RandomForest Classifier """
from sklearn.ensemble import RandomForestClassifier
clf2 = RandomForestClassifier(random_state=0)
clf2.fit(train_set_x, train_set_y)
RandomForestClassifier(random_state=0)

print(classification_report(train_set_y, clf2.predict(train_set_x)),"<--Train Set Evaluation")
print(classification_report(eval_set_y, clf2.predict(eval_set_x)),"<--Eval Set Evaluation")
print(classification_report(test_set_y, clf2.predict(test_set_x)),"<--Test Set Evaluation")
""" ___________________________________________________________"""
""" Classification3: SVM Classifier """

from sklearn import svm
clf3=svm.SVC()
clf3.fit(train_set_x, train_set_y)

print(classification_report(train_set_y, clf3.predict(train_set_x)),"<--Train Set Evaluation")
print(classification_report(eval_set_y, clf3.predict(eval_set_x)),"<--Eval Set Evaluation")
print(classification_report(test_set_y, clf3.predict(test_set_x)),"<--Test Set Evaluation")

""" ___________________________________________________________"""
""" Classification4: ِDecision Tree Classifier """


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
clf4 = tree.DecisionTreeClassifier()

clf4 = clf.fit(train_set_x, train_set_y)

print(classification_report(train_set_y, clf4.predict(train_set_x)),"<--Train Set Evaluation")
print(classification_report(eval_set_y, clf4.predict(eval_set_x)),"<--Eval Set Evaluation")
print(classification_report(test_set_y, clf4.predict(test_set_x)),"<--Test Set Evaluation")