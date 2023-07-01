import mlflow
from src.utils import transform_text
import pickle
import os

preprocessor_path=os.path.join('artifacts','vectorizer.pkl')
tfidf = pickle.load(open(preprocessor_path,'rb'))

runs = mlflow.search_runs(order_by=["start_time desc"], filter_string="")
latest_run_tags = runs['tags.mlflow.runName']
latest_model_run = runs[runs['tags.mlflow.runName'] == latest_run_tags].iloc[0]
model_run_id_ = latest_model_run['run_id']
model = mlflow.pyfunc.load_model(f'runs:/{model_run_id_}/MultiNomial')


def predict(input_sms):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    return result




