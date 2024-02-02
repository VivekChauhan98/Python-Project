#!/usr/bin/env python
# coding: utf-8

# In[66]:


from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree


# In[67]:


df = pd.read_csv('Crop_recommendation.csv')


# In[68]:


df.head()


# In[69]:


df.tail()


# In[70]:


print(f' Rows, Cols = {df.shape}')
print(f' Rows, Cols = {df.size}')


# In[71]:


df.columns


# In[72]:


df['label'].unique()


# In[73]:


#Making a Heatmap of the Dataset for the relation b/w the factors
linewidth = 2
linecolor = "red"

sns.heatmap(data = df.select_dtypes(include='number').corr(),annot=True,
           linewidth = linewidth,
           linecolor = linecolor)


# In[74]:


# from sklearn.model_selection import train_test_split
features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']

xtrain, xtest, ytrain, ytest = train_test_split(features,
                                                target,
                                                test_size = 0.2,
                                                random_state = None)


# In[75]:


#Defining a dict such that key = modelname and value = accuracy of the model

best_mode = {}


# In[ ]:





# # CROSS VALIDATION ACCURACY

# In[76]:


from sklearn.model_selection import cross_val_score

def cross_val_accuracy(model):
    
    score = cross_val_score(model, features, target, cv=5)
    #get the mean of each fold
    
    return score.mean()*100
    


# # Saving the Models

# In[77]:


import pickle

def save_model(model, modelname):
    #This function expects a model and a modelname(with .pkl extension)
    # Setting model path
    pkl_filename = 'C:\\Users\\Vivek Chauhan\\Downloads\\Soil detection'+modelname
    # Open the file to save as pkl file
    model_pkl = open(pkl_filename, 'wb')
    #dump model
    pickle.dump(model, model_pkl)
    # Close the pickle instances
    model_pkl.close()


# # LOGISTIC REGRESSION

# In[78]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Scale the data using StandardScaler
scaler = StandardScaler()
xtrain_scaled = scaler.fit_transform(xtrain)
xtest_scaled = scaler.transform(xtest)

# Create an instance of LogisticRegression
logreg = LogisticRegression(max_iter=1000)

# Fit the model to the scaled training data
logreg.fit(xtrain_scaled, ytrain)

# Use the model to make predictions on the test data
ypred = logreg.predict(xtest_scaled)

accuracy = metrics.accuracy_score(ytest, ypred)
print(f"Accuracy: {accuracy}",end='\n\n')

# Adding the model to our dictionary
best_model = (f"('Logistic Regression'): = { accuracy*100}")
print(best_model)


# In[79]:


from sklearn.pipeline import make_pipeline

#Create a pipeline with data scaling and logistic regression

pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

#Perform Cross-validation and calculate accuracy
score = cross_val_score(pipeline, features, target,cv=5, scoring = 'accuracy')

#Print the cross-validation accuracy scores

print("Cross-Validation Accuracy Scores:", score.mean()*100)


# In[80]:


save_model(logreg,'logreg.pkl')


# In[ ]:





# # DECISION TREE

# In[81]:


from sklearn.tree import DecisionTreeClassifier

DecisionTree = DecisionTreeClassifier(criterion = "entropy",
                                     random_state = None,
                                     max_depth = 5)
#Fitting the training set to create a model
DecisionTree.fit(xtrain,ytrain)

#Using test(x) to find y
ypred = DecisionTree.predict(xtest)

#Checking whether the predicted y for test(x) matches actual test(y)
accuracy =  metrics.accuracy_score(ypred, ytest)
print(f'Accuracy: {accuracy}',end ='\n\n')

#Adding the model to iur dictionary

best_model = (f'Decision Tree: {accuracy*100}') 
print(best_model)


# In[82]:


print(f'Cross Validation Accuracy {cross_val_accuracy(DecisionTree)}')


# In[83]:


save_model(DecisionTree,'DecisionTree.pkl')


# In[ ]:





# # RANDOM FOREST

# In[87]:


from sklearn.ensemble import RandomForestClassifier

RandomForest = RandomForestClassifier(n_estimators=20, random_state=None)

#Fitting the training set to create a model
RandomForest.fit(xtrain,ytrain)

#Using test(x) to find y
ypred  = RandomForest.predict(xtest)

accuracy = metrics.accuracy_score(ypred, ytest)
print(f'Accuracy: {accuracy}',end='\n\n')

#Adding the model to our dictionary
best_model = (f'Random Forest: {accuracy*100}')
print(best_model)


# In[88]:


print(f'Cross Validation Accuracy {cross_val_accuracy(RandomForest)}')


# In[90]:


save_model(RandomForest,'RandomForest.pkl')


# In[ ]:





# # Gaussian Naive BAYES

# In[94]:


from sklearn.naive_bayes import GaussianNB

NaiveBayes = GaussianNB()

# Fitting the training set to create a model
NaiveBayes.fit(xtrain,ytrain)
# Using test(x) to find y
ypred = NaiveBayes.predict(xtest)

accuracy = metrics.accuracy_score(ypred,ytest)
print(f"Accuracy: {accuracy}",end='\n\n')

# Adding the model to our dictionary
best_model = (f"Naive Bayes = {accuracy*100}")
print(best_model)


# In[96]:


print(f'Cross Validation Accuracy {cross_val_accuracy(NaiveBayes)}')


# In[97]:


save_model(NaiveBayes,'NaiveBayes.pkl')


# In[ ]:





# # Support Vector Machine

# In[103]:


from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

#fit scaler on Training data
norm = MinMaxScaler().fit(xtrain)
x_train_norm = norm.transform(xtrain)

#transform testing datas
x_test_norm = norm.transform(xtest)

SVM = SVC(kernel='poly', degree=3, C=1)
SVM.fit(x_train_norm,ytrain)
ypred = SVM.predict(x_test_norm)

accuracy = metrics.accuracy_score(ytest, ypred)
print(f'Accuracy: {accuracy}',end='\n\n')

#Adding the model to our dictionary
best_model = (f'SVM = {accuracy*100}')
print(best_model)


# In[ ]:





# # Comparing All Models

# In[109]:


# Assuming best_model is a dictionary with model names as keys and accuracies as values
best_model = {'LogReg':96.36363636, 'DecisionTree': 93.181818181, 'RandomForest': 99.31818181818,'GausianNaive bayes': 99.545454545454,'SVM':97.954545454}

# Extract model names and accuracies from the best_model dictionary
accuracies = list(best_model.values())
model_names = list(best_model.keys())

# Create a bar plot using seaborn
sns.barplot(x=accuracies, y=model_names)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Accuracies')
plt.show()


# In[111]:


sorted_items = sorted(best_model.items(), key=lambda x: x[1], reverse=True)

#PRint the keys and values in descending order

for key, value in sorted_items:
    print(f'{key}: {value}')


# In[112]:


max_key = max(best_model, key=best_model.get)
max_value  = best_model[max_key]

print(f'Key: {max_key}, Value: {max_value}')


# In[ ]:





# # USIING THE MODEL

# In[115]:


# Randomly select one row from the DataFrame
random_row = df.sample(n=1)

print(random_row)

# Extract the values of 'N', 'P', 'K', 'temperature', 'humidity', 'ph', and 'rainfall from the random row

random_values = random_row[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].values


# In[ ]:





# In[117]:


data = np.array(random_values)
data_with_feature_names = pd.DataFrame(data, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
prediction = NaiveBayes.predict(data_with_feature_names)
print(prediction)


# In[118]:


# Randomly select one row from the DataFrame
random_row = df.sample(n=1)

print(random_row)

# Extract the values of 'N', 'P', 'K', 'temperature', 'humidity', 'ph', and 'rainfall' from the random row
random_values = random_row[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].values


# In[119]:


data = np.array(random_values)
data_with_feature_names = pd.DataFrame(data, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

prediction = NaiveBayes.predict(data_with_feature_names)
print(prediction)


# In[ ]:




