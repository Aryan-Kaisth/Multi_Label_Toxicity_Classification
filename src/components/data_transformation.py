import os
import sys
from dataclasses import dataclass
from typing import Tuple, List, Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.main_utils import save_object, read_csv_file, read_yaml_file
from src.logger import logging
from src.exception import CustomException
from src.utils.text_utils import (
    to_lowercase,
    remove_urls,
    expand_contractions,
    remove_accents_diacritics,
    convert_emojis,
    remove_mentions,
    spacy_remove_punct_numbers_pipe,
    spacy_lemmatize_pipe,
    spacy_tokenize_pipe,
)

# Custom Text Preprocessor
class CustomPreprocessor(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible custom text preprocessor.
    Can be plugged into Pipelines, GridSearch, etc.
    """

    def __init__(self) -> None:
        pass

    def fit(self, X, y=None):
        """
        No learned parameters here
        """
        return self

    def transform(self, X):
        """
        Apply text cleaning pipeline to a 1D iterable of texts.
        """
        cleaned: List[str] = []

        try:
            for text in X:
                if not isinstance(text, str):
                    text = str(text)

                # Basic cleaning steps
                text = to_lowercase(text)
                text = remove_urls(text)
                text = expand_contractions(text)
                text = remove_accents_diacritics(text)
                text = convert_emojis(text)
                text = remove_mentions(text)

                # spaCy-based pipeline
                tokens = spacy_tokenize_pipe(text)               # tokenize
                tokens = spacy_remove_punct_numbers_pipe(tokens) # remove punct & numbers
                tokens = spacy_lemmatize_pipe(tokens)            # lemmatize

                cleaned.append(" ".join(tokens))

            return cleaned

        except Exception as e:
            logging.error("[ERROR] Failure inside CustomPreprocessor.transform")
            raise CustomException(e, sys)


# Config

@dataclass
class DataTransformationConfig:
    """
    Holds paths and config required during data transformation.
    """
    preprocessor_file_path: str = os.path.join( "artifacts", "data_transformation", "custom_preprocessor.pkl")
    schema_path: str = os.path.join("config", "schema.yaml")


# Main Transformation Class
class DataTransformation:
    """
    Orchestrates feature cleaning, text preprocessing, and train/test splitting
    for model-ready inputs.
    """

    def __init__(self, config: DataTransformationConfig = DataTransformationConfig()) -> None:
        self._config = config

        # Ensure artifact directory exists
        os.makedirs(os.path.dirname(self._config.preprocessor_file_path), exist_ok=True)
        logging.info(
            f"[INIT] Data transformation artifacts directory ensured: "
            f"{os.path.dirname(self._config.preprocessor_file_path)}"
        )

        # Load schema
        self._schema: dict[str, Any] = read_yaml_file(self._config.schema_path)
        logging.info(f"[INIT] Loaded schema from: {self._config.schema_path}")

        # Read schema lists
        self._target_cols: List[str] = self._schema.get("target_cols", [])
        self._text_cols: List[str] = self._schema.get("text_cols", [])
        self._drop_cols: Any = self._schema.get("drop_cols", [])

        if isinstance(self._drop_cols, str):
            self._drop_cols = [self._drop_cols]
        elif self._drop_cols is None:
            self._drop_cols = []

        logging.info(
            f"[INIT] target_cols={self._target_cols}, "
            f"text_cols={self._text_cols}, drop_cols={self._drop_cols}"
        )

    # ----- Private Helpers -----

    def _feature_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic feature cleaning:
        - drop columns (id, etc.)
        - remove duplicates
        - drop rows with nulls
        """
        try:
            logging.info("[CLEAN] Feature cleaning started")

            # Drop specified columns
            if self._drop_cols:
                existing_to_drop = [c for c in self._drop_cols if c in df.columns]
                df = df.drop(columns=existing_to_drop, errors="ignore")
                logging.info(f"[CLEAN] Dropped columns: {existing_to_drop}")

            # Drop duplicates
            before = df.shape[0]
            df = df.drop_duplicates().reset_index(drop=True)
            after_dupes = df.shape[0]

            # Drop null values
            df = df.dropna().reset_index(drop=True)
            after_na = df.shape[0]

            logging.info(
                f"[CLEAN] Removed {before - after_dupes} duplicates and "
                f"{after_dupes - after_na} rows with NaNs."
            )
            return df

        except Exception as e:
            logging.error("[ERROR] Feature cleaning failed.")
            raise CustomException(e, sys)

    def _text_preprocessing(
        self, df: pd.DataFrame, preprocessor: CustomPreprocessor
    ) -> pd.DataFrame:
        """
        Apply CustomPreprocessor to all text columns defined in schema.
        """
        try:
            logging.info("[TEXT] Text preprocessing started")

            for col in self._text_cols:
                if col not in df.columns:
                    logging.warning(
                        f"[TEXT] Column '{col}' missing in dataframe, skipping text preprocessing for it."
                    )
                    continue

                logging.info(f"[TEXT] Preprocessing column: {col}")
                # Ensure we pass a 1D array/Series to the transformer
                df[col] = preprocessor.transform(df[col].astype(str).values)

            logging.info("[TEXT] Text preprocessing completed")
            return df

        except Exception as e:
            logging.error("[ERROR] Text preprocessing step failed.")
            raise CustomException(e, sys)

    def _split_features_targets(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame into X (features) and y (targets) using schema columns.
        """
        try:
            logging.info("[SPLIT] Splitting into features and targets")

            missing_targets = [c for c in self._target_cols if c not in df.columns]
            if missing_targets:
                logging.warning(
                    f"[SPLIT] Missing target columns in dataframe: {missing_targets}"
                )

            X = df[self._text_cols].copy()
            y = df[self._target_cols].copy()

            logging.info(
                f"[SPLIT] X shape: {X.shape}, y shape: {y.shape}"
            )
            return X, y

        except Exception as e:
            logging.error("[ERROR] Feature/target split failed.")
            raise CustomException(e, sys)

    def _save_preprocessor(self, preprocessor: CustomPreprocessor) -> None:
        """
        Persist the fitted preprocessor for later inference.
        """
        try:
            save_object(self._config.preprocessor_file_path, preprocessor)
            logging.info(
                f"[SAVE] Preprocessor object saved at: {self._config.preprocessor_file_path}"
            )
        except Exception as e:
            logging.error("[ERROR] Failed to save preprocessor object.")
            raise CustomException(e, sys)

    # ----- Public Orchestrator -----

    def initiate_data_transformation(
        self, train_path: str, test_path: str
    ) -> Tuple[Any, Any, Any, Any]:
        """
        Public method to orchestrate full transformation:
        - Load train/test CSVs
        - Clean features
        - Apply text preprocessing
        - Split into X/y
        - Save preprocessor

        Returns:
            X_train, X_test, y_train, y_test
        """
        logging.info("==== DATA TRANSFORMATION STARTED ====")

        try:
            # Load raw train and test data
            train_df = read_csv_file(train_path)
            test_df = read_csv_file(test_path)
            logging.info(
                f"[LOAD] Train shape: {train_df.shape}, Test shape: {test_df.shape}"
            )

            # Cleaning
            train_df = self._feature_cleaning(train_df)
            test_df = self._feature_cleaning(test_df)

            # Instantiate and apply preprocessor
            preprocessor = CustomPreprocessor()

            train_df = self._text_preprocessing(train_df, preprocessor)
            test_df = self._text_preprocessing(test_df, preprocessor)

            # Split into features/targets
            X_train_df, y_train_df = self._split_features_targets(train_df)
            X_test_df, y_test_df = self._split_features_targets(test_df)

            # Save preprocessor
            self._save_preprocessor(preprocessor)

            logging.info("==== DATA TRANSFORMATION COMPLETED SUCCESSFULLY ====")

            # Return numpy arrays for model training + preprocessor path
            return (
                X_train_df.values,
                X_test_df.values,
                y_train_df.values,
                y_test_df.values,
            )

        except Exception as e:
            logging.error("[FATAL] Data transformation pipeline failed.")
            raise CustomException(e, sys)
        

# ---- Testing ----
if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion, DataIngestionConfig
    from src.components.data_transformation import DataTransformation  # make sure it's imported

    # Initialize data ingestion
    ingest_config = DataIngestionConfig()
    data_ingestion = DataIngestion(config=ingest_config)

    # Perform ingestion to get train and test file paths
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    # Initialize DataTransformation
    transformer = DataTransformation()

    # Run the data transformation
    X_train_transformed, X_test_transformed, y_train, y_test = transformer.initiate_data_transformation(
        train_path=train_data_path,
        test_path=test_data_path
    )

    # Quick sanity checks
    print("✅ Transformed X_train shape:", X_train_transformed.shape)
    print("✅ Transformed X_test shape:", X_test_transformed.shape)
    print("✅ y_train shape:", y_train.shape)
    print("✅ y_test shape:", y_test.shape)

