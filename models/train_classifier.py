import sys
import re
import pickle
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'stopwords', 'wordnet'])

def load_data(database_filepath):
    """
    Load data from an SQLite database and return feature and target variables.

    Args:
    database_filepath (str): Filepath for the SQLite database containing the cleaned data.

    Returns:
    X (DataFrame): Features dataframe containing messages for training.
    Y (DataFrame): Target dataframe containing the category labels for each message.
    category_names (list of str): List of category names.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterData', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns.tolist()
    
    return X, Y, category_names


def tokenize(text):
    """
    Process text data: tokenization, normalization, removal of stopwords, and lemmatization.

    Args:
    text (str): The raw message text to be tokenized.

    Returns:
    tokens (list of str): Cleaned and tokenized text data.
    """
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_regex, 'urlplaceholder', text)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    tokens = word_tokenize(text)
    tokens = [tk for tk in tokens if tk not in stopwords.words("english")]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens


def build_model():
    """
    Build a machine learning pipeline that includes text vectorization, 
    TF-IDF transformation, and a multi-output classifier.

    Returns:
    pipeline (Pipeline): A scikit-learn Pipeline object for training the model.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

#     parameters = {
#         'clf__estimator__n_estimators': [50, 100],
#     }

#     cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of the trained model using test data and print out classification reports.

    Args:
    model: Trained model to be evaluated.
    X_test (DataFrame): Test features.
    Y_test (DataFrame): True labels for the test data.
    category_names (list of str): Names of the target categories.

    Returns:
    None
    """
    Y_pred = model.predict(X_test)

    for i, column in enumerate(category_names):
        print(f"Category: {column}")
        print(classification_report(Y_test[column], Y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Save the trained model as a pickle file.

    Args:
    model: The trained machine learning model to be saved.
    model_filepath (str): The filepath for saving the pickle file.

    Returns:
    None
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


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