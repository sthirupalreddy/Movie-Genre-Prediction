import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    """Preprocesses text data: lowercase, remove punctuation, stopwords, lemmatization."""
    if not isinstance(text, str):
        return ""  # Handle non-string inputs

    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

def train_and_evaluate_model(df, text_column, genre_column, model_type='naive_bayes'):
    """Trains and evaluates a genre prediction model."""

    df['processed_text'] = df[text_column].apply(preprocess_text)
    X = df['processed_text']
    y = df[genre_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tfidf_vectorizer = TfidfVectorizer(max_features=5000) # limit to top 5000 features
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    if model_type == 'naive_bayes':
        model = MultinomialNB()
    elif model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000)
    elif model_type == 'svm':
        model = SVC(kernel='linear') # Linear kernel is often good for text data.
    else:
        raise ValueError("Invalid model_type. Choose from 'naive_bayes', 'logistic_regression', or 'svm'.")

    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0) #handles cases where a class has no predictions.

    print(f"Model: {model_type}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)

    return model, tfidf_vectorizer # returning the model and vectorizer so they can be used later.

# Example usage (replace with your dataset)
# Assuming your dataset is in a CSV file called 'movies.csv' with columns 'plot' and 'genre'
try:
    df = pd.read_csv('movie.csv')
    #Clean up genres.
    df = df.dropna(subset=['Genre', 'Runtime']) # remove any rows with missing genre or plot data.
    df['Genre'] = df['Genre'].apply(lambda x: x.split(',')[0].strip() if isinstance(x, str) else str(x)) # Take only the first genre.
    df = df[df['Genre'].str.len() > 0] # remove rows with empty genres.

    # Example calls. You can change model_type.
    naive_bayes_model, naive_bayes_vectorizer = train_and_evaluate_model(df, 'Runtime', 'Genre', model_type='naive_bayes')
    logistic_regression_model, logistic_regression_vectorizer = train_and_evaluate_model(df, 'Runtime', 'Genre', model_type='logistic_regression')
    svm_model, svm_vectorizer = train_and_evaluate_model(df, 'Runtime', 'Genre', model_type='svm')

    #Example prediction:
    example_plot = "A detective investigates a series of mysterious murders in a dark city."
    processed_example = preprocess_text(example_plot)
    example_tfidf = svm_vectorizer.transform([processed_example]) #use the vectorizer from the model you want to use.
    prediction = svm_model.predict(example_tfidf)
    print(f"Predicted genre for '{example_plot}': {prediction[0]}")

except FileNotFoundError:
    print("Error: movies.csv not found. Please provide your movie dataset.")
except KeyError as e:
    print(f"Error: Column '{e.args[0]}' not found in the CSV. Please check your column names.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")