# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset.

2.Separate input features (X) and target variable (Y).

3.Split the dataset into training and testing data.

4.Train the Logistic Regression model using training data.

5.Predict the output using test data.

6.Calculate accuracy and other performance measures. 

## Program:
```
/*
Program to implement Logistic Regression for classifying food choices based on nutritional information.
Developed by: Varoodhini.M
RegisterNumber: 212225220118 
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('food_items (1).csv')

print('Name:Varoodhini.M')
print('Reg.No: 212225220118')
print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())
X_raw=df.iloc[:,:-1]
y_raw=df.iloc[:,-1:]
scaler=MinMaxScaler()
X=scaler.fit_transform(X_raw)
label_encoder=LabelEncoder()
y=label_encoder.fit_transform(y_raw.values.ravel())
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=123)
penalty='l2'
multi_class='multinomial'
solver='lbfgs'
max_iter=1000
l2_model = LogisticRegression(random_state=123, penalty=penalty, multi_class=multi_class, solver=solver, max_iter=max_iter)
l2_model.fit(X_train,y_train)
y_pred=l2_model.predict(X_test)
print('Name: ')
print('Reg.No: ')
print("\nModel Evaluation: ")
print("Accuracy: ",accuracy_score(y_test,y_pred))
print("\nClassification Report: ")
print(classification_report(y_test,y_pred))
conf_matrix=confusion_matrix(y_test,y_pred)
print(conf_matrix)
print('Name: Varoodhini.M')
print('Reg No:212225220118 ')


```

## Output:

<img width="766" height="610" alt="image" src="https://github.com/user-attachments/assets/7ed56c6a-1ee6-4020-abe8-c2de362c8af3" />


<img width="576" height="391" alt="image" src="https://github.com/user-attachments/assets/a8b39710-210c-40f7-9ed4-26bf38276432" />



<img width="217" height="129" alt="image" src="https://github.com/user-attachments/assets/0cf181c1-48c5-42fa-9f9a-7ebaa9c4cce5" />


## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
