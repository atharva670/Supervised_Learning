from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,mean_absolute_error,mean_squared_error,r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pandas as pd
import math
import numpy as np
# Load data
df = pd.read_csv('train.csv')
#Decision Tree


le_x = LabelEncoder()
le_y = LabelEncoder()


X = le_x.fit_transform(df['Item_Type'])
y = le_y.fit_transform(df['Outlet_Type'])



X = pd.DataFrame(X)


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.5, random_state=0)


model = DecisionTreeClassifier()
model.fit(X_train, Y_train)


y_pred = model.predict(X_test)
Y_test_decoded = le_y.inverse_transform(Y_test)
y_pred_decoded = le_y.inverse_transform(y_pred)



df1 = pd.DataFrame({
    'Actual': Y_test_decoded,
    'Predicted': y_pred_decoded
})

print(df1)
print(confusion_matrix(Y_test,y_pred))

print("Accuracy:", accuracy_score(y_pred, Y_test)*100,'%')
#Logistic regression

le_x1 = LabelEncoder()
le_y1 = LabelEncoder()


X1 = le_x1.fit_transform(df['Item_Type'])
y1 = le_y1.fit_transform(df['Outlet_Type'])



X1=pd.DataFrame(X1)


X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, y1, test_size=0.5, random_state=0)


model1 = LogisticRegression()
model1.fit(X1_train, Y1_train)


y_pred1 = model1.predict(X1_test)
Y1_test_decoded = le_y1.inverse_transform(Y1_test)
y1_pred_decoded = le_y1.inverse_transform(y_pred1)



df2 = pd.DataFrame({
    'Actual': Y1_test_decoded,
    'Predicted': y1_pred_decoded
})

print(df2)
print(confusion_matrix(Y1_test,y_pred1))

print("Accuracy:", accuracy_score(y_pred1, Y1_test)*100,'%')
#Support Vector Machines
obj1=LabelEncoder()
obj2=LabelEncoder()
fat=obj1.fit_transform(df['Item_Fat_Content'])
type1=obj2.fit_transform(df['Item_Type'])
fat=pd.DataFrame(fat)
x_train,x_test,y_train,y_test=train_test_split(fat,type1,test_size=0.5,random_state=0)
mod=SVC(kernel='rbf')
mod.fit(x_train,y_train)
predicted=mod.predict(x_test)
new_test=obj2.inverse_transform(predicted)
new_test1=obj2.inverse_transform(y_test)
dr=pd.DataFrame({'Actual':new_test,
                 'Predicted':new_test1
                })
print(dr)
print(confusion_matrix(new_test1,new_test))

print("Accuracy:", accuracy_score(new_test1, new_test)*100,'%')
#Linear Regression
pc=pd.read_csv('Placement.csv')
inp=pd.DataFrame(pc['cgpa'])
print(pc.columns)
out=pc['package ']
x_tr,x_ts,y_tr,y_ts=train_test_split(inp,out,test_size=0.5,random_state=0)
mo=LinearRegression()
mo.fit(x_tr,y_tr)
p=mo.predict(x_ts)
dh=pd.DataFrame({'Actual':y_ts.values,
                 'Predicted':p})
print(dh)
MAE=mean_absolute_error(y_ts,p)
MSE=mean_squared_error(y_ts,p)
RMSE=math.sqrt(MSE)
R2=r2_score(y_ts,p)
print('Mean Absolute Error',MAE,'\n','Mean Squared Error',MSE,'\n','Root of Mean Squared Error',RMSE,'\n','R2 Score',R2)
w=float(input('Enter CGPI='))
w1=np.array([[w]])
print('Package=',mo.predict(w1))













