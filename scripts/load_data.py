import os
import psycopg2
import pandas as pd 
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

load_dotenv()

# Fetch database connection parameters from environment variables
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

def load_data_from_postgres(query):
    """
    Connects to the PostgreSQL database and loads data based on the provided SQL query.

    :param query: SQL query to execute.
    :return: DataFrame containing the results of the query.
    """
    try:
        # Establish a connection to the database
        connection = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )

        # Load data using pandas
        df = pd.read_sql_query(query, connection)

        # Close the database connection
        connection.close()

        return df
        

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def check_missing_values(df):
    missing_values = df.isnull().sum()
    missing_values_df = missing_values[missing_values > 0].reset_index()
    missing_values_df.columns = ['column_name', 'missing_count']
    return missing_values_df

def missing_percentage(df):
    missing_percentatges = df.isnull().mean() * 100 
    column_with_missing_values = missing_percentatges[missing_percentatges > 30].index.tolist()
    return column_with_missing_values

def save_df_to_postgres(df, table_name):
    # Create an engine instance
    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

    # Save DataFrame to PostgreSQL
    df.to_sql(table_name, engine, if_exists='replace', index=False)

def check_duplicates(df):
    """
    Check for duplicate rows in a DataFrame and return a summary.
    
    Parameters:
    - df: Pandas DataFrame
    
    Returns:
    - DataFrame or str: Summary of duplicate rows or success message
    """
    # Find duplicate rows
    duplicates = df[df.duplicated(keep='first')]
    
    if duplicates.empty:
        return "Success: No duplicated values."
    else:
        # Get the first column name
        first_column_name = df.columns[0]
        
        # Get the first column value from the duplicated rows
        duplicates_summary = duplicates[[first_column_name]].copy()
        duplicates_summary['Number of Duplicates'] = duplicates.groupby(first_column_name)[first_column_name].transform('count')
        
        # Drop duplicate rows from the summary
        duplicates_summary.drop_duplicates(inplace=True)
        
        return duplicates_summary
    
def get_numeric_columns(df):
    """
    Get a list of column names with numeric data types from a DataFrame.
    
    Parameters:
    - df: Pandas DataFrame
    
    Returns:
    - numeric_columns: List of column names with numeric data types
    """
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    return numeric_columns
    
def check_numeric_anomalies(df, column, lower_bound=None, upper_bound=None):
    """
    Check for numeric anomalies in a specific column of a DataFrame and return a summary.
    
    Parameters:
    - df: Pandas DataFrame
    - column: The specific column to check
    - lower_bound: Lower bound for numeric anomalies (optional)
    - upper_bound: Upper bound for numeric anomalies (optional)
    
    Returns:
    - str or DataFrame: Success message or summary of anomalies
    """
    if df[column].dtype not in ['int64', 'float64']:
        return f"Error: Column {column} is not numeric."
    
    if lower_bound is not None and upper_bound is not None:
        anomalies = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    else:
        anomalies = df[~df[column].apply(lambda x: isinstance(x, (int, float)))]
    
    if anomalies.empty:
        return "Success: No anomalies detected."
    else:
        anomalies_summary = pd.DataFrame({
            'Column Name': [column],
            'Number of Anomalies': [len(anomalies)]
        })
        return anomalies_summary