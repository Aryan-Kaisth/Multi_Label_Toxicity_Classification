# src/pipelines/training_pipeline.py

import os
import sys
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class TrainingPipelineConfig:
    """
    Configuration holder for all pipeline-level settings.
    (Future scope: artifact paths, orchestration flags, etc.)
    """
    enable_debug: bool = False  # for future use


class TrainingPipeline:
    """
    End-to-end ML training pipeline.

    Workflow:
        1. Data Ingestion
        2. Data Transformation
        3. Model Training
    """
    # PRIVATE STEPS

    def _run_data_ingestion(self):
        """Runs data ingestion and returns train/test file paths."""
        try:
            logging.info("[STEP] Starting Data Ingestion...")

            ingestion = DataIngestion(DataIngestionConfig())

            train_path, test_path = ingestion.initiate_data_ingestion()

            logging.info(f"[STEP] Data Ingestion completed. "
                         f"Train path: {train_path}, Test path: {test_path}")

            return train_path, test_path

        except Exception as e:
            logging.error("[ERROR] Data Ingestion failed.")
            raise CustomException(e, sys)

    def _run_data_transformation(self, train_path: str, test_path: str):
        """Runs data transformation and returns processed arrays."""
        try:
            logging.info("[STEP] Starting Data Transformation...")

            transformer = DataTransformation()

            X_train, X_test, y_train, y_test = transformer.initiate_data_transformation(
                train_path=train_path,
                test_path=test_path
            )

            logging.info(f"[STEP] Data Transformation completed. "
                         f"X_train: {X_train.shape}, X_test: {X_test.shape}")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error("[ERROR] Data Transformation failed.")
            raise CustomException(e, sys)

    def _run_model_training(self, X_train, X_test, y_train, y_test):
        """Runs model training and evaluation."""
        try:
            logging.info("[STEP] Starting Model Training...")

            trainer = ModelTrainer()

            model = trainer.initiate_model_trainer(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test
            )

            logging.info("[STEP] Model Training completed successfully.")
            return model

        except Exception as e:
            logging.error("[ERROR] Model Training failed.")
            raise CustomException(e, sys)

    # PUBLIC PIPELINE RUNNER
    def run_pipeline(self):
        """
        Public method to orchestrate the entire ML pipeline.
        Returns the trained model instance.
        """
        try:
            logging.info("===== ML TRAINING PIPELINE STARTED =====")

            # Data Ingestion
            train_path, test_path = self._run_data_ingestion()

            # Data Transformation
            X_train, X_test, y_train, y_test = self._run_data_transformation(
                train_path=train_path,
                test_path=test_path
            )

            # Model Training
            model = self._run_model_training(X_train, X_test, y_train, y_test)

            logging.info("===== ML TRAINING PIPELINE COMPLETED SUCCESSFULLY =====")
            return model

        except Exception as e:
            logging.error("[FATAL] Training pipeline execution failed.")
            raise CustomException(e, sys)
