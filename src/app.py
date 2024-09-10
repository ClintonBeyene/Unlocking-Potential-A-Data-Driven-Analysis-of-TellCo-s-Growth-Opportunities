import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sys
import os
sys.path.append(os.path.abspath('../scripts'))
from load_data import load_data_from_postgres, check_missing_values

# Load the model from the file
model = pickle.load(open('model.pkl', 'rb'))

# Define a function to make predictions
def make_predictions(X):
    return model.predict(X)

# Define a function to track the model's performance
def track_performance(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    print(f'MSE: {mse:.2f}')

# Define the main function
def main():
    # Load the data
    query = "SELECT * FROM satisfaction_analysis;"
    df = load_data_from_postgres(query)
    
    if df is None:
        print("DataFrame is None. Check data loading process.")
        return
    else:
        print("DataFrame loaded successfully.")
        print(df.head())  # Print the first few rows of the DataFrame for verification

    # Check for missing values
    missing_values_df = check_missing_values(df)
    print("Missing values in the DataFrame:")
    print(missing_values_df)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.2, random_state=42)

    # Make predictions on the testing data
    y_pred = make_predictions(X_test)

    # Track the model's performance
    track_performance(y_test, y_pred)

if __name__ == '__main__':
    query = "SELECT * FROM satisfaction_analysis LIMIT 10;"
    df = load_data_from_postgres(query)
    if df is not None:
        print(df.head())
    else:
        print("Failed to load data")
