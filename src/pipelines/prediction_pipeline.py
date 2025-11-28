# src/pipelines/prediction_pipeline.py

import os
import sys
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from typing import Dict, Any

from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import load_object
from keras.models import load_model


@dataclass
class PredictionPipelineConfig:
    """Stores paths for model + preprocessor."""
    preprocessor_path: str = os.path.join("artifacts", "data_transformation", "custom_preprocessor.pkl")
    model_path: str = os.path.join("artifacts", "model_trainer", "bi_lstm.keras")


class PredictionPipeline:
    """
    Handles complete prediction workflow:
      - Load preprocessor
      - Load trained model
      - Validate input
      - Preprocess text
      - Generate prediction
    """

    def __init__(self, config: PredictionPipelineConfig = PredictionPipelineConfig()):
        """
        Initialize PredictionPipeline with required components.
        """
        self._config = config

        try:
            logging.info("[INIT] Initializing PredictionPipeline...")

            self._preprocessor = self._load_preprocessor()
            self._model = self._load_model()

            logging.info("[INIT] PredictionPipeline initialized successfully.")

        except Exception as e:
            logging.error("[ERROR] PredictionPipeline initialization failed.")
            raise CustomException(e, sys)

    # PRIVATE METHODS
    def _load_preprocessor(self):
        """Load preprocessing object."""
        try:
            logging.info(f"[LOAD] Loading preprocessor: {self._config.preprocessor_path}")
            preprocessor = load_object(self._config.preprocessor_path)
            logging.info("[LOAD] Preprocessor loaded successfully.")
            return preprocessor

        except Exception as e:
            logging.error("[ERROR] Failed to load preprocessor.")
            raise CustomException(e, sys)

    def _load_model(self):
        """Load trained TF/Keras model."""
        try:
            logging.info(f"[LOAD] Loading model: {self._config.model_path}")
            model = load_model(self._config.model_path)
            logging.info("[LOAD] Model loaded successfully.")
            return model

        except Exception as e:
            logging.error("[ERROR] Failed to load model.")
            raise CustomException(e, sys)

    def _apply_preprocessing(self, text: str):
        """Run custom preprocessing pipeline."""
        try:
            logging.info("[PROCESS] Applying preprocessing...")
            transformed = self._preprocessor.transform([text])
            logging.info(f"[PROCESS] Preprocessing complete: {transformed}")
            return transformed

        except Exception as e:
            logging.error("[ERROR] Preprocessing failed.")
            raise CustomException(e, sys)

    def _run_model_prediction(self, transformed_text: Any):
        """Run model prediction on preprocessed text."""
        try:
            logging.info("[PREDICT] Generating model predictions...")

            preds = self._model.predict(
                tf.constant(transformed_text),
                verbose=0
            )

            logging.info(f"[PREDICT] Raw model output: {preds}")
            return preds

        except Exception as e:
            logging.error("[ERROR] Prediction failed.")
            raise CustomException(e, sys)

    # PUBLIC METHOD
    def predict(self, text: str) -> Dict[str, float]:
        """
        Public method to generate prediction from raw text.
        Returns probabilities for each toxic label.
        """
        try:
            logging.info(f"[INPUT] Received text for prediction: {text}")

            # Preprocess
            transformed = self._apply_preprocessing(text)

            # Predict
            preds = self._run_model_prediction([transformed])
            preds = preds[0]  # flatten

            logging.info(f"[SUCCESS] Final prediction: {preds}")

            # Format output
            return {
                "toxic": float(preds[0]),
                "severe_toxic": float(preds[1]),
                "obscene": float(preds[2]),
                "threat": float(preds[3]),
                "insult": float(preds[4]),
                "identity_hate": float(preds[5])
            }

        except Exception as e:
            logging.error("[FATAL] Prediction pipeline execution failed.")
            raise CustomException(e, sys)
