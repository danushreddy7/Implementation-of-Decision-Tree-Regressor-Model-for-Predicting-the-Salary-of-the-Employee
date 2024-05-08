# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: T DANUSH REDDY
RegisterNumber:212223040029
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])  
*/
```

## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)
# Data.Head()
![image](https://github.com/danushreddy7/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/149035740/92a6bc62-ace6-4250-b3e4-b23e922c7238)
# Data.info()
![image](https://github.com/danushreddy7/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/149035740/9354fd8c-6e57-4938-bc9e-456310a98427)
# isnull() and sum()
![image](https://github.com/danushreddy7/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/149035740/ef43d52b-9d2f-4786-ab85-ee0bf388c9a8)
Data.Head() for salary:
![318781337-ca9e940c-44ee-42eb-bb35-df88b05fd65c](https://github.com/danushreddy7/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/149035740/01a759c1-b257-4906-ab29-a0395c6715d3)
# MSE Value:
![318781651-8e759f19-abd2-4e41-94c0-3edbccc6104c](https://github.com/danushreddy7/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/149035740/63838909-48a1-4399-ba92-ca9b24005cbc)
# R2 VALUE:
![318781930-d8f2a1a3-5083-4d77-8c8b-8d795cbd8c0b](https://github.com/danushreddy7/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/149035740/79c44636-440b-4f97-a780-90cd8cdb540d)
# Data Prediction:
![318782341-0c020faa-c263-41eb-86cd-b9c1e569376a](https://github.com/danushreddy7/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/149035740/90e3f665-e41d-43db-a9b0-e619ab6822e8)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
