import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    """
    Load Messages Data with Categories Function
    
    Arguments:
        messages_filepath -> Path to the CSV file containing messages
        categories_filepath -> Path to the CSV file containing categories
    Returns:
        df -> Combined data containing messages and categories
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages , categories, on = 'id', how = 'inner' )
    
    return df


def clean_data(df):
    
    
    """
    Cleans the combined dataframe for use by ML model
    
    Arguments:
    df pandas_dataframe: Merged dataframe returned from load_data() function
    
    Returns:
    df pandas dataframe, Cleaned data to be used by ML model
    
    """
    categories = df['categories'].str.split(pat = ';' , expand= True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    row = row.values.tolist()

    # use this row to extract a list of new column names for categories.
    # up to the second to last character of each string with slicing

    category_colnames = []
    for n in row:
        lists1 = n[:-2]
        category_colnames.append(lists1)
        
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:].astype(np.int64)
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories],  axis=1) 
    
    # drop duplicates
    df = df.drop_duplicates()
    df = df[df['related'] != 2]
    
    return df


def save_data(df, database_filename):
    
    """
    Saves cleaned data to an SQL database
    
    Arguments:
    
    df pandas_dataframe: Cleaned data returned from clean_data() function
    database_file_name: File path of SQL Database into which the cleaned
    data is to be saved
    
    Returns:
    None
    """    
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('DisasterResponseTable', engine, index=False , if_exists='replace')


def main():
    
    
    """
    Main function which will kick off the data processing functions. There are three primary actions taken by this function:
        1) Load Messages Data with Categories
        2) Clean Categories Data
        3) Save Data to SQLite Database
    """

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()