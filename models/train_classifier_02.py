import sys
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk import word_tokenize
import nltk
import re
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from custom_transformer import StartingVerbExtractor
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
import pickle
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """
    This function reads SQL database through sqlalchemy and loads 
    database table into Pandas dataframe extracting feature values and target values
    input:
        Database file name and path 
    Output:
        X = Feature values (Text) 
        Y = Target values as result of classification process
        col_names = Target values (Y) column names 
    """
    
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    
    # Create datframe by quering database
    df = pd.read_sql("SELECT * from messages", engine)
    
    # Feature selection
    X = df['message']
    
    # Choosing column names for multiobjective classification
    category_names=df.drop(['id','message','original','genre'], axis=1).columns
    
    # Target values to predict
    Y =df[category_names] 
    
    return X, Y, category_names


def tokenize(text):
    # Normalize text
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]"," ",text)
    
    # Tokenize text
    words = word_tokenize(text)
    
    # Remove stop words
    tokens = [w for w in words if w not in stopwords.words('english')]    
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip() # [WordNetLemmatizer().lemmatize(w) for w in tokens]
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    """
    This function use SKLEARN pipeline library to create classifier model
    input:
        NONE
    output:
        classifier model
    """
    #
    pipeline = Pipeline([
        ('features', FeatureUnion([
           ('text_pipeline', Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),('tfidf', TfidfTransformer())])),
            ('verb', StartingVerbExtractor())])),
         ('clf', RandomForestClassifier())
    ])
    
    # hyerparameters for grid to search within
#    parameters = [{'clf__bootstrap': [False, True],
#          'clf__bootstrap': [False, True],
#          'clf__n_estimators': [80,90, 100, 110, 130],
#          'clf__max_features': [0.6, 0.65, 0.7, 0.73, 0.7500000000000001, 0.78, 0.8],
#          'clf__min_samples_leaf': [10, 12, 14],
#          'clf__min_samples_split': [3, 5, 7]
#         }
#     ]

    
    
    
    parameters = {
    'clf__n_estimators': [50, 100, 200],
    'clf__min_samples_split': [2, 3, 4],
    }


    # Final model ready to be applied on dataset
    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model



def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model on the test data and print the result
    input:
        model  = trained classifier model
        X_test = testing features (Unseen data)
        Y_test = true values to compare with prediction on unseen test cases
        category_names = column name of Y_test data
    output:
        print model prediction accuracy on test data
    """
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))
    pass   


def save_model(model, model_filepath):
    """
    Saving trained model on on disk to be load when required.
    input:
        model = trained classifier
        model_filepath = file path and name to store model on disk
        
    """
    # using pickle to store trained classifier
    pickle.dump(model,open(model_filepath,'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()