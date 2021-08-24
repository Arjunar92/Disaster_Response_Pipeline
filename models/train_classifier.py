import sys
# import libraries
import pandas as pd  #To Import data
import numpy as np   #For Numerical analysis
import matplotlib.pyplot as plt #Data plotting and visualization
import seaborn as sns #For Data visualization
import re
import nltk
import pickle
nltk.download('averaged_perceptron_tagger')
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator,TransformerMixin

import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

def load_data(database_filepath):
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql("DisasterResponseTable", engine)
    df = df[df['related'] != 2]
    X = df['message']
    y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])
    
    return X, y , category_names


def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex , text)
    
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
        
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline_improved = Pipeline([
        
    ('features', FeatureUnion([
        
        ('text_pipeline', Pipeline([
            
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        
        ]))   
    ])),
        
        
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
        
             
    ])
        
        
        
    
    parameters_improved = {
    'clf__estimator__n_estimators': [10, 20, 40]
    }
    
    cv_improved = GridSearchCV(pipeline_improved, param_grid=parameters_improved , scoring='f1_micro')
    
    return cv_improved

def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)
    
    for i in range(len(category_names)):
        print('Category: {} '.format(category_names[i]))
        print(classification_report(y_test.iloc[:, i].values, y_pred[:, i]))
        print('Accuracy {}\n\n'.format(accuracy_score(y_test.iloc[:, i].values, y_pred[:, i])))
    
    # Print classification report on test data
    print(classification_report(y_test.iloc[:, 0:].values, np.array([x[0:] for x in y_pred]), target_names = category_names))
    print('Accuracy {}\n\n'.format(accuracy_score(y_test.iloc[:, i].values, y_pred[:, i])))

def save_model(model, model_filepath):
    pickle.dump(model, open('model.pkl', 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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