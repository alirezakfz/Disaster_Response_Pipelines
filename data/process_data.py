import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
        
    """
    Loads data from messages and categories datasets, returns merged dataframe
    
    input:
          File path for CSV files.
    output:
            Merged dataframe 
    
    """
    #reading files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #merges dataframes with ID as keys and return it
    return messages.merge(categories, on='id')


def clean_data(df):
    """
    Task of this function is:
        a) Split the values in the categories column on the ; character so that each value becomes a separate column.
        b) Use the first row of categories dataframe to create column names for the categories data.
        c) Rename columns of categories with new column names.
        d) Convert category values to just numbers 0 or 1
            i) Iterate through the category columns in df to keep only the last character of each string (the 1 or 0).
            ii) Convert column from string to numeric
    Input:
        Merged dataframe from messages and categories
    Output:
        Cleaned and ready dataframe to apply ML algorithms and prediction model
    """
    
    # seperating first row of the dataframe to be used for column names
    category_colnames = [col.split('-')[0] for col in df.categories[0].split(';')]
    
    # create a dataframe of the 36 individual category columns
    categories = pd.Series(df.categories).str.split(';',n=36, expand=True)
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = [str(col).split('-')[1] for col in categories[column]]
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    #related column contains some values of '2', re-assign them as '1'
    categories.loc[categories['related']==2,'related']=1
    
    # drop the original categories column from `df`
    df.drop(['categories'], axis = 1, inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis = 1)
    
    # drop duplicates
    df.drop_duplicates(keep='first', inplace=True)

    return df


def save_data(df, database_filename):
    """
    This function saves the dataframe into SQL database
    
    input: 
         Dataframe as df and name for the database as database_filename
    
    """
        
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False, if_exists='replace')
    pass


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