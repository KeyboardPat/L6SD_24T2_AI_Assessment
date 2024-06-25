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
def load_data(data):
    data = pd.read_excel('Net_Worth_Data.xlsx')
    return data

def preprocess_data(data):
    x= data.drop(['Client Name', 'Client e-mail', 'Profession', 'Education', 'Country', 'Net Worth'], axis=1)
    y= data['Net Worth']
    sc = MinMaxScaler()
    x_scaled = sc.fit_transform(x)
    sc1 = MinMaxScaler()
    y_reshape = y.values.reshape(-1, 1)
    y_scaled = sc1.fit_transform(y_reshape)
    return x_scaled, y_scaled, sc, sc1

def split_data(x_scaled, y_scaled):
    train_test_split(x_scaled, y_scaled, test_size = 0.2, random_state = 42)

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
x= data.drop(['Client Name', 'Client e-mail', 'Profession', 'Education', 'Country'], axis=1)
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

#Print first few rows of scaled input dataset
print("First 5 rows of scaled input dataset\n", x_scaled[:5], y_scaled[:5])
#Print last few rows of scaled output dataset
print("Last 5 rows of scaled input dataset\n", x_scaled[5:], y_scaled[5:])

#Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size = 0.2, random_state = 42)

#Import and initialize AI models
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor

lr = LinearRegression()
ls = Lasso()
Ri = Ridge
svr = SVR()
rf = RandomForestRegressor()
gbr = GradientBoostingRegressor()
Ada = AdaBoostRegressor
ExtTr = ExtraTreesRegressor
xgb = XGBRegressor()
dtr = DecisionTreeRegressor

#Train models using training data
lr.fit(x_train, y_train)
ls.fit(x_train, y_train)
Ri.fit(x_train, y_train)
svr.fit(x_train, y_train)
rf.fit(x_train, y_train)
gbr.fit(x_train, y_train)
Ada.fit(x_train, y_train)
ExtTr.fit(x_train, y_train)
xgb.fit(x_train, y_train)
dtr.fit(x_train, y_train)

#Prediction on test data
lr_preds = lr.predict(x_test)
ls_preds = ls.predict(x_test)
Ri_preds = Ri.predict(x_test)
svr_preds = svr.predict(x_test)
rf_preds = rf.predict(x_test)
gbr_preds = gbr.predict(x_test)
Ada_preds = Ada.predict(x_test)
ExtTr_preds = ExtTr.predict(x_test)
xgb_preds = xgb.predict(x_test)
dtr_preds = dtr.predict(x_test)

#Evaluate model performance
#RMSE is a measure of the differences between the predicted values by the model and the actual values
lr_rmse = mean_squared_error(y_test, lr_preds, squared = False)
ls_rmse = mean_squared_error(y_test, ls_preds, squared=False)
Ri_rmse = mean_squared_error(y_test, Ri_preds, squared=False)
svr_rmse = mean_squared_error(y_test, svr_preds, squared = False)
rf_rmse = mean_squared_error(y_test, rf_preds, squared = False)
gbr_rmse = mean_squared_error(y_test, gbr_preds, squared = False)
Ada_rmse = mean_squared_error(y_test, Ada_preds, squared=False)
ExtTr_rmse = mean_squared_error(y_test, ExtTr_preds, squared = False)
xgb_rmse = mean_squared_error(y_test, xgb_preds, squared = False)
dtr_rmse = mean_squared_error(y_test, dtr_preds, squared=False)

# Evaluate most accurate model
models = [lr, ls, Ri, svr, rf, gbr, Ada, ExtTr, xgb, dtr]
model_rmse = [lr_rmse, ls_rmse, Ri_rmse, svr_rmse, rf_rmse, gbr_rmse, Ada_rmse, ExtTr_rmse, xgb_rmse, dtr_rmse]

best_model_index = model_rmse.index(min(model_rmse))
best_model_object = models[best_model_index]

#Display evaluation results
print(f"Linear Regression RMSE: {lr_rmse}")
print(f"Lasso RMSE: {ls_rmse}")
print(f"Ridge RMSE: {Ri_rmse}")
print(f"SVR RMSE: {svr_rmse}")
print(f"RF RMSE: {rf_rmse}")
print(f"Gradient Boosting Regressor RMSE: {gbr_rmse}")
print(f"Ada RMSE: {Ada_rmse}")
print(f"Extra Tree RMSE: {ExtTr_rmse}")
print(f"XGB Regressor RMSE: {xgb_rmse}")
print(f"DTR RMSE: {dtr_rmse}")

#Visualize model results by creating a bar chart
#Visualize RMSE values
plt.figure(figsize=(10, 11))
bars = plt.bar(model_rmse.keys(), model_rmse.values(), color=['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'black', 'grey'])

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.00001, round(yval, 5), ha='center', va='bottom', fontsize=10)

plt.xlabel('Models')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title('Model RMSE Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#visualize_rmse(model_rmse)

#Save the model
dump(best_model_object, "networth_model.joblib")

#Load the model
loaded_model = load("networth_model.joblib")

#Gather user inputs
gender = int(input("Enter gender (0 for female, 1 for male): "))
age = int(input("Enter age: "))
income = float(input("Enter annual salary: "))
credit_card_debt = float(input("Enter credit card debt: "))
healthcare_cost = int(input("Enter 1 for healthcare cost: "))
inherited_amount = float(input("Enter inherited amount: "))
stocks = float(input("Enter stocks: "))
bonds = float(input("Enter bonds: "))
mutual_funds = float(input("Enter mutual funds: "))
EFTs = float(input("Enter EFTS: "))
REITs = float(input("Enter REITs: "))

#Use model to make predictions based on user input
x_test1 = sc.transform([[gender, age, income, credit_card_debt, healthcare_cost, inherited_amount, stocks, bonds, mutual_funds, EFTs, REITs]])
#Predict on new test data
pred_value = loaded_model.predict(x_test1)
print(pred_value)
print("Predicted Net Worth based on input: ", sc1.inverse_transform(pred_value))