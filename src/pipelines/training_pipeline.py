from src.components.data_ingestion import data_ingestion
from src.components.data_transformation import data_transformation
from src.components.model_trainer import model_training

if __name__ == "__main__":
    data_path = data_ingestion()
    X,y = data_transformation(data_path)
    model_training(X,y)

