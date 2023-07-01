import mlflow
import mlflow.sklearn
import pickle

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score


def model_training(X,y):
    with mlflow.start_run():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

        model = MultinomialNB()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accu_score = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy_score", accu_score)
        mlflow.log_metric("precision_score", precision)

        mlflow.sklearn.log_model(model, "MultiNomial")
        print('Success')



