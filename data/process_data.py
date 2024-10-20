# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
     """
    Load and merge message and category datasets.

    Args:
    messages_filepath (str): Filepath for the CSV file containing messages data.
    categories_filepath (str): Filepath for the CSV file containing categories data.

    Returns:
    DataFrame: A merged DataFrame of messages and categories datasets.
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """
    Clean the combined dataset by splitting categories into individual columns,
    converting category values to binary, and removing duplicates.

    Args:
    df (DataFrame): Combined dataset of messages and categories.

    Returns:
    DataFrame: Cleaned dataset with separate category columns, numeric values, and no duplicates.
    """
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    
    # Convert category values to just numbers 0 or 1.¶
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # Replace categories column in df with new category columns.
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # Remove duplicates.
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """
    Save the cleaned dataset into an SQLite database.

    Args:
    df (DataFrame): The cleaned dataset to be saved.
    database_filename (str): The filename for the SQLite database.

    Returns:
    None
    """
    # Create SQLite engine
    engine = create_engine('sqlite:///' + database_filename)

    # Save the dataframe to the SQLite database
    df.to_sql('DisasterData', engine, index=False, if_exists='replace')


def main():
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