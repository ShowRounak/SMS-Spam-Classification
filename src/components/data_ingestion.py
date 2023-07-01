import os
import pandas as pd
import numpy as np
import mlflow
from sklearn.preprocessing import LabelEncoder

def data_ingestion():
    df = pd.read_csv('notebook/spam.csv', encoding='latin-1')
    df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True)
    df.rename(columns={'v1':"target",'v2':"text"},inplace=True)

    encoder = LabelEncoder()
    df['target'] = encoder.fit_transform(df['target'])

    df = df.drop_duplicates(keep='first')

    raw_data_path = os.path.join('artifacts','data.csv')
    os.makedirs(os.path.dirname(raw_data_path),exist_ok=True)
    df.to_csv(raw_data_path,index=False)
    print('data ingest completed successfully')

    return raw_data_path