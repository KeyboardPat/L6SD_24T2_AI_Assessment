import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from joblib import dump, load

#Import the dataset
data = pd.read_excel('Net_Worth_Data.xlsx')

#Display the first 5 rows
print("First 5 rows of dataset\n", data.head())
#Display the last 5 rows
print("Last 5 rows of dataset\n", data.tail())
#Determine the shape of the dataset
print("Number of rows and columns\n", data.shape)
print("Number of rows\n", data.shape[0])
print("Number of columns\n", data.shape[1])
#Display the concise summary of the dataset
print("Concise summary of data:\n")
print(data.info())
#Check the null values in the dataset
print("To check null values:\n")
print(data.isnull())
print(data.isnull().sum())

#Create the input dataset from the original dataset by dropping the irrelevant
#store input variables in x
x= data.drop(['Client Name', 'Client e-mail', 'Profession', 'Education', 'Country', 'Healthcare Cost'], axis=1)
print(x)

#create the output dataset from the original dataset
#store output variable in y
y= data['Net Worth']
print(y)

#Transform input dataset into percentage based weighted between 0 and 1
sc = MinMaxScaler()
x_scaled = sc.fit_transform(x)
print(x_scaled)

#Transform output dataset into percentage based weighted between 0 and 1
sc1 = MinMaxScaler()
y_reshape = y.values.reshape(-1, 1)
y_scaled = sc1.fit_transform(y_reshape)
print(y_scaled)

#Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size = 0.2, random_state = 42)

#Import and initialize AI models
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

lr = LinearRegression()
svm = SVR()
rf = RandomForestRegressor()
gbr = GradientBoostingRegressor()
xgb = XGBRegressor()

#Train models using training data
lr.fit(x_train, y_train)
svm.fit(x_train, y_train)
rf.fit(x_train, y_train)
gbr.fit(x_train, y_train)
xgb.fit(x_train, y_train)

#Prediction on test data
lr_preds = lr.predict(x_test)
svm_preds = svm.predict(x_test)
rf_preds = rf.predict(x_test)
gbr_preds = gbr.predict(x_test)
xgb_preds = xgb.predict(x_test)

#Evaluate model performance
#RMSE is a measure of the differences between the predicted values by the model and the actual values
lr_rmse = mean_squared_error(y_test, lr_preds, squared = False)
svm_rmse = mean_squared_error(y_test, svm_preds, squared = False)
rf_rmse = mean_squared_error(y_test, rf_preds, squared = False)
gbr_rmse = mean_squared_error(y_test, gbr_preds, squared = False)
xgb_rmse = mean_squared_error(y_test, xgb_preds, squared = False)

# Evaluate most accurate model
models = [lr, svm, rf, gbr, xgb]
model_rmse = [lr_rmse, svm_rmse, rf_rmse, gbr_rmse, xgb_rmse]

best_model_index = model_rmse.index(min(model_rmse))
best_model_object = models[best_model_index]

#Display evaluation results
print(f"Linear Regression RMSE: {lr_rmse}")
print(f"SVM RMSE: {svm_rmse}")
print(f"RF RMSE: {rf_rmse}")
print(f"Gradient Boosting Regressor RMSE: {gbr_rmse}")
print(f"XGB Regressor RMSE: {xgb_rmse}")

#Visualize model results by creating a bar chart
#Visualize RMSE values
def visualize_rmse(model_rmse):
    plt.figure(figsize=(10, 7))
    bars = plt.bar(model_rmse.keys(), model_rmse.values(), color=['blue', 'green', 'red', 'purple', 'orange'])
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.00001, round(yval, 5), ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Models')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title('Model RMSE Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

visualize_rmse(model_rmse)

#Save the model
dump(best_model_object, "networth_model.joblib")

#Load the model
loaded_model = load("networth_model.joblib")