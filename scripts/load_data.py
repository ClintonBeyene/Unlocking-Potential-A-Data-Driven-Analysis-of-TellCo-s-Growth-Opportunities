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