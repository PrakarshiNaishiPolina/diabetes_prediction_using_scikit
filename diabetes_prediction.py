from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import pandas as pd


# load the diabetes dataset
diabetes=load_diabetes()

# converting into dataframe
data=pd.DataFrame(data=diabetes.data,columns=diabetes.feature_names)
data['target']=diabetes.target

print(data.head())

# features and target

X= data.drop(columns='target')
y=data['target']

# split data into training and testing sets

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

print("Training data shape: ",X_train.shape)
print("Testing data shape: ",X_test.shape)

# train a regression model

model=LinearRegression()

# train the model
model.fit(X_train,y_train)

# get coefficients and intercept
print("Coefficients: ",model.coef_)
print("Intercept: ",model.intercept_)


# make predictions and evaluate using mse,mae,r2


y_pred=model.predict(X_test)

# evaluate 
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")





















