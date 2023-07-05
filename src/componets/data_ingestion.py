import os
import sys

sys.path.append("src/error")
from error import CustomException
from logger import logging

import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from componets.data_transformation import DataTransformation
from componets.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Started initialization of data ingesation")
        try:
            df = pd.read_csv("notebook\data\stud.csv")
            os.makedirs(
                (os.path.dirname(self.ingestion_config.train_data_path)), exist_ok=True
            )

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            train, test = train_test_split(df, test_size=0.2, random_state=42)

            train.to_csv(self.ingestion_config.train_data_path, header=True, index=False)
            test.to_csv(self.ingestion_config.test_data_path, header=True, index=False)

            logging.info("Train and test are initialized and saved in the directory")

            return (
                self.ingestion_config.test_data_path,
                self.ingestion_config.train_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    test, train = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, path = data_transformation.initiate_data_transformer(train, test)

    model_trainer = ModelTrainer()
    score = model_trainer.initiate_model_trainer(train_arr=train_arr, test_arr=test_arr)

    print(score)
