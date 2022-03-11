

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

import xgboost as xg
from sklearn import linear_model
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from google.colab import drive
drive.mount('/content/drive')

# loading in dataset
numbers = pd.read_csv('/content/drive/MyDrive/train.csv')
print(numbers.head())
print(numbers.shape)

sns.countplot(numbers['label'], palette='icefire')

some_number = numbers.iloc[4,1:]
some_number = some_number.values.reshape(28,28)
plt.imshow(some_number,cmap='gray')

some_number = numbers.iloc[67,1:]
some_number = some_number.values.reshape(28,28)
plt.imshow(some_number,cmap='gray')

x = numbers.iloc[:, 1:]
y = numbers.iloc[:,0]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.5, random_state=101)

print("X_train: ", x_train.shape)
print("Y_train: ", y_train.shape)
print("X_test: ", x_test.shape)
print("Y_test: ", y_test.shape)

# Running through different models 

names = []
training_times = []
results_test = []
results_train = []
models = []

models.append(('SVM-Linear', SVC(kernel='linear')))
models.append(('SVM-Non Linear', SVC(kernel='rbf')))
models.append(('Decision Tree', DecisionTreeClassifier()))
models.append(('Random Forest', RandomForestClassifier()))
models.append(('XG Boost', xg.XGBClassifier()))
models.append(('Ada Boost', AdaBoostClassifier()))

for name, model in models:
  names.append(name)
  start_time = time.time()
  model.fit(x_train,y_train)
  end_time = time.time()
  time_taken = end_time - start_time
  time_taken = round(time_taken,2)
  training_times.append(time_taken)  

  train_acc = 100*model.score(x_train,y_train)
  test_acc = 100*model.score(x_test,y_test)

  results_train.append(train_acc)
  results_test.append(test_acc)

  print("Time taken to train using", name, "Classifier: ", time_taken, "s")
  print("Train Accuracy of ", name, " model: ", round(train_acc,5), "%")
  print("Test Accuracy of ", name, " model: ", round(test_acc,5), "%")
  print("--------------------------------------------------")

# plotting 
fig = plt.figure(figsize=(10,7))
plt.bar(names, training_times)
plt.title('Training time comparision')
plt.xlabel('Models', fontweight ='bold', fontsize = 15)
plt.ylabel('Time (s)', fontweight ='bold', fontsize = 15)
plt.show


fig = plt.figure(figsize=(10,7))
plt.title('Training Accuracy comparision')
plt.bar(names, results_train)
plt.xlabel('Models', fontweight ='bold', fontsize = 15)
plt.ylabel('Accuracy (%)', fontweight ='bold', fontsize = 15)
plt.show

fig = plt.figure(figsize=(10,7))
plt.title('Testing Accuracy comparision')
plt.bar(names, results_test)
plt.xlabel('Models', fontweight ='bold', fontsize = 15)
plt.ylabel('Accuracy (%)', fontweight ='bold', fontsize = 15)
plt.show

fig = plt.figure(figsize=(10,7))
plt.title('Testing Acc v/s Training Acc')
plt.plot(results_train, results_test)
plt.xlabel('Models', fontweight ='bold', fontsize = 15)
plt.ylabel('Accuracy (%)', fontweight ='bold', fontsize = 15)
plt.show

import autosklearn.classification

automl = autosklearn.classification.AutoSklearnClassifier()
automl.fit(x_train, y_train)
pred = automl.predict(x_test)
acc = 100*metrics.accuracy_score(y_true=y_test, y_pred=pred)
print(acc)

print(automl.sprint_statistics())
print(automl.show_models())

results_df = pd.DataFrame(automl.cv_results_)
results_df.to_csv('results_mnist.csv')

