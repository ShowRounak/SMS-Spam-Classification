import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils import transform_text
import pickle

def data_transformation(data_path):
    
    df = pd.read_csv(data_path)

    df['transformed_text'] = df['text'].apply(transform_text)

    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df['transformed_text']).toarray()
    y = df['target'].values

    print('data TRANSFORMATION completed successfully')

    vectorizer_data_path = os.path.join('artifacts', 'vectorizer.pkl')
    os.makedirs(os.path.dirname(vectorizer_data_path),exist_ok=True)

    pickle.dump(tfidf,open(vectorizer_data_path,'wb'))
    
    return X, y