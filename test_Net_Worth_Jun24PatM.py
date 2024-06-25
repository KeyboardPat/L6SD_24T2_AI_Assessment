import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Net_Worth_Jun24PatM import load_data, preprocess_data, train_test_split
import pandas as pd
import numpy as np

#Test case to ensure the correct shape of data
def test_shape_of_data():
    data = load_data('Net_Worth_Data.xlsx')
    X_scaled, Y_scaled, _, _ = preprocess_data(data)
    
    #expected number of columns
    assert X_scaled.shape[1] == 11, "Expected 11 features in the X data after preprocessing."
    assert Y_scaled.shape[1] == 1, "Expected Y data to have a single column."
    #expected number of rows
    assert X_scaled.shape[0] == 500, "Expected 5 features in the X data after preprocessing."
    assert Y_scaled.shape[0] == 500

#Test case to ensure the correct columns for Input
def test_columns_X():
    data = load_data('Net_Worth_Data.xlsx')
    X, _, _, _ = preprocess_data, (load_data)
    input_columns = ['Gender', 'Age', 'Income', 'Credit Card Debt', 'Inherited Amount', 'Stocks', 'Bonds', 'Mutual Funds', 'EFTs', 'REITs']
    # Convert the NumPy array to a DataFrame
    X_df = pd.DataFrame(X, columns=input_columns)
    # Check if X_df is a DataFrame
    assert isinstance(X_df, pd.DataFrame)
    # Check that the columns have been dropped for X
    assert "Client Name" not in X_df.columns
    assert "Client e-mail" not in X_df.columns
    assert "Profession" not in X_df.columns
    assert "Education" not in X_df.columns
    assert "Country" not in X_df.columns
    assert "Healthcare Cost" not in X_df.columns

#Test case to ensure the correct column for output
def test_columns_Y():
    Data = load_data('Net_Worth_Jun24PatM.xlsx')
    _, Y, _, _ = preprocess_data(load_data)
    # Convert the NumPy array to a DataFrame
    Y_df = pd.DataFrame(Y, columns=['Net Worth'])
    
    # Check if Y_df is a DataFrame and has the correct column name
    assert isinstance(Y_df, pd.DataFrame)
    assert Y_df.columns == 'Net Worth'
    assert 'Net Worth' in Y_df.columns