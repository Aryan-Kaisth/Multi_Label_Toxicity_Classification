# src/components/model_trainer.py
import os, sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import save_object
from keras.layers import LSTM, Dense, BatchNormalization, LayerNormalization, Dropout, Activation, Input
from keras.optimizers import Nadam
from keras.models import Sequential
import pandas as pd
import numpy as np
import gensim.downloader as api

@dataclass
class ModelTrainerConfig:
    model_file_path: str = os.path.join("artifacts", "model_trainer", "gru.keras")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        os.makedirs(os.path.dirname(self.config.model_file_path), exist_ok=True)
        logging.info(f"ModelTrainer initialized. Model will be saved at {self.config.model_file_path}")

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        """
        Trains a LSTM Deep learning model and evaluates it.
        Saves the trained model as a keras file.
        """
        try:
            logging.info("Model training started")
            from keras.layers import TextVectorization, Embedding

            vectorizer = TextVectorization(
                max_tokens=50000,
                output_sequence_length=70,
                output_mode='int'
            )
            vectorizer.adapt(X_train)

            glove200_model = api.load('glove-twitter-200')
            vocab = vectorizer.get_vocabulary()
            vocab_size = len(vocab)
            embedding_dim = glove200_model.vector_size

            embedding_matrix = np.zeros((vocab_size, embedding_dim))

            for i, word in enumerate(vocab):
                if word in glove200_model:
                    embedding_matrix[i] = glove200_model[word]

            embedding_layer = Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                weights=[embedding_matrix],
                trainable=True,
                mask_zero=True
            )
#------------------------------------------------------------------------------
            model = Sequential(name='aryan_gru')
            model.add(vectorizer)
            model.add(embedding_layer)
            model.add(LSTM(units=64))
            model.add(Activation('tanh'))
            model.add(LayerNormalization())
            model.add(Dense(32, kernel_initializer='he_normal'))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.8))
            model.add(Dense(16, kernel_initializer='he_normal'))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(Dense(units=1, activation='sigmoid'))
            logging.info(f"Summary of the Model: {model.summary(show_trainable=True, line_length=115)}")

            model.compile(optimizer=Nadam(), loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, batch_size=512, epochs=10, validation_split=0.2)
            accuracy = model.evaluate(X_test, y_test)
            logging.info(f"Model accuracy and error: {accuracy}")
            
            # Save model
            model.save(self.config.model_file_path)
            logging.info(f"Trained model saved at: {self.config.model_file_path}")
            
            logging.info("Model training Completed")
            return model

        except Exception as e:
            logging.error("Error in model training")
            raise CustomException(e, sys)
        


# ---- Testing ----
if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion, DataIngestionConfig
    from src.components.data_transformation import DataTransformation

    # Paths to train and test data
    ingest_config = DataIngestionConfig()
    data_ingestion = DataIngestion(config=ingest_config)
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    # Initialize the transformer
    transformer = DataTransformation()

    # Run the data transformation
    X_train_transformed, X_test_transformed, y_train, y_test = transformer.initiate_data_transformation(
        train_path=train_data_path,
        test_path=test_data_path
    )

    model = ModelTrainer()
    model.initiate_model_trainer(X_train_transformed, X_test_transformed, y_train, y_test)