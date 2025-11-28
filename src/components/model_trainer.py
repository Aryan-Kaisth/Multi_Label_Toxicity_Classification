# src/components/model_trainer.py

import os
import sys
import numpy as np
import gensim.downloader as api
from dataclasses import dataclass
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import save_object  # if unused, you can remove

from tensorflow import keras
from keras.models import Sequential
from keras.layers import (
    TextVectorization,
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    BatchNormalization,
    LayerNormalization,
    Dropout,
    Activation
)
from keras.optimizers import Nadam
from keras_cv.losses import FocalLoss  # type: ignore


@dataclass
class ModelTrainerConfig:
    """
    Configuration for model training artifacts.
    """
    model_file_path: str = os.path.join(
        "artifacts", "model_trainer", "bilstm_model.keras"
    )


class ModelTrainer:
    """
    Handles the full deep-learning training workflow:
    - Build TextVectorizer
    - Load GloVe embeddings
    - Construct Embedding Matrix
    - Build BiLSTM Model
    - Compile & Train Model with Focal Loss
    - Save Model
    """

    def __init__(self, config: ModelTrainerConfig = ModelTrainerConfig()):
        self._config = config

        try:
            os.makedirs(os.path.dirname(self._config.model_file_path), exist_ok=True)
            logging.info(
                f"[INIT] ModelTrainer initialized. "
                f"Model will be saved at: {self._config.model_file_path}"
            )
        except Exception as e:
            raise CustomException(e, sys)

    def _build_vectorizer(self, X_train):
        """
        Create and adapt the TextVectorization layer.

        Parameters
        ----------
        X_train : array-like
            Training text data.

        Returns
        -------
        TextVectorization
            Adapted text vectorizer.
        """
        try:
            logging.info("[BUILD] Initializing TextVectorization layer...")

            vectorizer = TextVectorization(
                max_tokens=50_000,
                output_sequence_length=120,
                output_mode="int",
            )
            vectorizer.adapt(X_train)

            logging.info("[BUILD] TextVectorization layer built & adapted.")
            return vectorizer

        except Exception as e:
            logging.error("[ERROR] Failed to build TextVectorizer.")
            raise CustomException(e, sys)

    def _load_glove(self):
        """
        Load GloVe pretrained word embeddings.

        Returns
        -------
        gensim.models.KeyedVectors
            Loaded GloVe embeddings.
        """
        try:
            logging.info("[LOAD] Loading GloVe embeddings: glove-wiki-gigaword-100")
            glove_model = api.load("glove-wiki-gigaword-100")
            logging.info("[LOAD] GloVe embeddings loaded successfully.")
            return glove_model

        except Exception as e:
            logging.error("[ERROR] Failed to load GloVe embeddings.")
            raise CustomException(e, sys)

    def _build_embedding_matrix(self, vectorizer, glove_model):
        """
        Create embedding matrix aligned to vectorizer vocabulary.

        Parameters
        ----------
        vectorizer : TextVectorization
            Adapted Keras TextVectorization layer.
        glove_model : gensim.models.KeyedVectors
            Pretrained GloVe embeddings.

        Returns
        -------
        np.ndarray
            Embedding matrix of shape (vocab_size, embedding_dim).
        """
        try:
            logging.info("[BUILD] Building embedding matrix...")

            vocab = vectorizer.get_vocabulary()
            vocab_size = len(vocab)
            embedding_dim = glove_model.vector_size

            word_index = {word: idx for idx, word in enumerate(vocab)}
            embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)

            match_count = 0
            for word, idx in word_index.items():
                if word in glove_model.key_to_index:
                    embedding_matrix[idx] = glove_model[word]
                    match_count += 1

            match_ratio = match_count / vocab_size if vocab_size > 0 else 0.0
            logging.info(
                f"[BUILD] Embedding matrix built. "
                f"Matched {match_count}/{vocab_size} words "
                f"({match_ratio:.2%} coverage)."
            )

            return embedding_matrix

        except Exception as e:
            logging.error("[ERROR] Failed to build embedding matrix.")
            raise CustomException(e, sys)

    def _build_model(self, vectorizer, embedding_matrix):
        """
        Construct the BiLSTM model architecture.

        Parameters
        ----------
        vectorizer : TextVectorization
            Adapted text vectorizer layer.
        embedding_matrix : np.ndarray
            Pretrained embedding matrix.

        Returns
        -------
        keras.Model
            Compiled BiLSTM model.
        """
        try:
            logging.info("[BUILD] Constructing BiLSTM model...")

            vocab_size, embedding_dim = embedding_matrix.shape

            embedding_layer = Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                weights=[embedding_matrix],
                trainable=True,
                mask_zero=True,
                name="pretrained_embedding",
            )

            model = Sequential(name="aryan_bilstm_1")
            model.add(vectorizer)
            model.add(embedding_layer)

            # Recurrent backbone
            model.add(
                Bidirectional(
                    LSTM(128, return_sequences=False),
                    name="bilstm_1"
                )
            )
            model.add(LayerNormalization(name="layer_norm_1"))

            # Dense stack
            model.add(Dense(64, kernel_initializer="he_normal", name="dense_1"))
            model.add(BatchNormalization())
            model.add(Activation("relu"))
            model.add(Dropout(0.5))

            model.add(Dense(32, kernel_initializer="he_normal", name="dense_2"))
            model.add(BatchNormalization())
            model.add(Activation("relu"))
            model.add(Dropout(0.3))

            # Output layer (6 labels)
            model.add(Dense(units=6, activation="sigmoid", name="output"))

            model.summary(print_fn=lambda x: logging.info(x))
            logging.info("[BUILD] BiLSTM model constructed.")
            return model

        except Exception as e:
            logging.error("[ERROR] Failed to build BiLSTM model.")
            raise CustomException(e, sys)

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        """
        Run the full model training pipeline.

        Steps
        -----
        - Build text vectorizer
        - Load GloVe embeddings and build embedding matrix
        - Build BiLSTM model
        - Compile with Focal Loss
        - Train and validate
        - Save trained model

        Parameters
        ----------
        X_train : pandas.Series or array-like
            Training texts.
        X_test : pandas.Series or array-like
            Test texts (not used in training but may be used later).
        y_train : pandas.DataFrame or array-like
            Training labels (multi-label).
        y_test : pandas.DataFrame or array-like
            Test labels (multi-label).

        Returns
        -------
        keras.Model
            Trained Keras model.
        """
        try:
            logging.info("===== MODEL TRAINING STARTED =====")

            # 1. Build vectorizer
            vectorizer = self._build_vectorizer(X_train)

            # 2. Load pretrained GloVe embeddings
            glove_model = self._load_glove()

            # 3. Build embedding matrix
            embedding_matrix = self._build_embedding_matrix(vectorizer, glove_model)

            # 4. Build model
            model = self._build_model(vectorizer, embedding_matrix)

            # 5. Compile model with Focal Loss
            loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
            model.compile(
                optimizer=Nadam(learning_rate=3e-4),
                loss=loss_fn,
                metrics=[
                    keras.metrics.AUC(multi_label=True, name="auc_roc"),
                    keras.metrics.Precision(name="precision"),
                    keras.metrics.Recall(name="recall"),
                ],
            )

            logging.info("[TRAIN] Starting model.fit()...")

            model.fit(
                X_train,
                y_train,
                epochs=10,
                batch_size=128,
                validation_split=0.2,
                verbose=1,
            )

            # 6. Save model
            model.save(self._config.model_file_path)
            logging.info(
                f"[SAVE] Model successfully saved to: {self._config.model_file_path}"
            )

            logging.info("===== MODEL TRAINING COMPLETED SUCCESSFULLY =====")
            return model

        except Exception as e:
            logging.error("[FATAL] Model training failed.")
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