#Common preparing tools
 
 
## Importing the libraries
"""
 
 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
 
"""## Importing the dataset"""
 
 
dataset = pd.read_csv('Copy of Data.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
 
 
"""## Splitting the dataset into the Training set and Test set"""
 
 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
 
 
"""## Feature Scaling"""
 
 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
 
 
"""# Training the models
 
 
###Naive Bayes
"""
 
 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
 
 
classifier_Bayes = GaussianNB()
classifier_Bayes.fit(X_train, y_train)
y_pred = classifier_Bayes.predict(X_test)
cm_Bayes = confusion_matrix(y_test, y_pred)
acc_score_bayes = accuracy_score(y_test, y_pred)
 
 
"""### Decision Tree"""
 
 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
 
 
classifier_tree = DecisionTreeClassifier(criterion = 'entropy')
classifier_tree.fit(X_train, y_train)
y_pred = classifier_tree.predict(X_test)
cm_tree = confusion_matrix(y_test, y_pred)
acc_score_tree = accuracy_score(y_test, y_pred)
 
 
"""### Logistic regression"""
 
 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
 
 
classifier_log_reg = LogisticRegression()
classifier_log_reg.fit(X_train, y_train)
y_pred = classifier_log_reg.predict(X_test)
cm_log_reg = confusion_matrix(y_test, y_pred)
acc_score_log_reg = accuracy_score(y_test, y_pred)
 
 
"""### K-NN"""
 
 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
 
 
classifier_knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_knn.fit(X_train, y_train)
y_pred = classifier_knn.predict(X_test)
cm_knn = confusion_matrix(y_test, y_pred)
acc_score_knn = accuracy_score(y_test, y_pred)
 
 
"""### Forest"""
 
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
 
 
classifier_forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
classifier_forest.fit(X_train, y_train)
y_pred = classifier_forest.predict(X_test)
cm_forest = confusion_matrix(y_test, y_pred)
acc_score_forest = accuracy_score(y_test, y_pred)
 
 
"""###SVM linear"""
 
 
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
 
 
classifier_SVM_lin = SVC(kernel = 'linear')
classifier_SVM_lin.fit(X_train, y_train)
y_pred = classifier_SVM_lin.predict(X_test)
cm_SVM_lin = confusion_matrix(y_test, y_pred)
acc_score_SVM_lin = accuracy_score(y_test, y_pred)
 
 
"""### SVM kernel"""
 
 
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
 
 
classifier_SVM_rbf = SVC(kernel = 'rbf')
classifier_SVM_rbf.fit(X_train, y_train)
y_pred = classifier_SVM_rbf.predict(X_test)
cm_SVM_rbf = confusion_matrix(y_test, y_pred)
acc_score_SVM_rbf = accuracy_score(y_test, y_pred)
 
 
"""# Results"""
 
 
print(f"score for Naive Bayes: {acc_score_bayes} % accuracy")
print(f"score for Decision Tree: {acc_score_tree} % accuracy")
print(f"score for Logistic Regression: {acc_score_log_reg} % accuracy")
print(f"score for K_NN: {acc_score_knn} % accuracy")
print(f"score for Forest: {acc_score_forest} % accuracy")
print(f"score for SVM linear: {acc_score_SVM_lin} % accuracy")
print(f"score for SVM kernel: {acc_score_SVM_rbf} % accuracy")
