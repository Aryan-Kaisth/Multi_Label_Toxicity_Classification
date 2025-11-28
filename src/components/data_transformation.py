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
    spacy_lemmatize_pipe, spacy_remove_punct_numbers_pipe, spacy_tokenize_pipe
)

# Custom Text Preprocessor
class CustomPreprocessor(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible text preprocessor integrating:
    - regex cleaning
    - contractions
    - emoji → text
    - accents removal
    - spaCy cleaning, lemmatization & tokenization
    """

    def __init__(self) -> None:
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Apply full text cleaning pipeline to a 1D iterable of texts.
        """
        try:
            logging.info("[PIPELINE] Starting CustomPreprocessor...")

            step1 = [to_lowercase(t) for t in X]
            step2 = [remove_urls(t) for t in step1]
            step3 = [expand_contractions(t) for t in step2]
            step4 = [remove_accents_diacritics(t) for t in step3]
            step5 = [convert_emojis(t) for t in step4]
            step6 = [remove_mentions(t) for t in step5]

            spacy_cleaned = spacy_remove_punct_numbers_pipe(step6)
            lemmatized = spacy_lemmatize_pipe(spacy_cleaned)
            tokenized = spacy_tokenize_pipe(lemmatized)
            final_texts = [" ".join(tokens) for tokens in tokenized]

            logging.info("[PIPELINE] CustomPreprocessor completed successfully.")
            return final_texts

        except Exception as e:
            logging.error("[ERROR] Failure inside CustomPreprocessor.transform")
            raise CustomException(e, sys)

# Config
@dataclass
class DataTransformationConfig:
    preprocessor_file_path: str = os.path.join("artifacts", "data_transformation", "custom_preprocessor.pkl")
    schema_path: str = os.path.join("config", "schema.yaml")


# Main Transformation Class
class DataTransformation:

    def __init__(self, config: DataTransformationConfig = DataTransformationConfig()) -> None:
        self._config = config

        os.makedirs(os.path.dirname(self._config.preprocessor_file_path), exist_ok=True)
        logging.info(f"[INIT] Data transformation artifacts directory ensured.")

        self._schema: dict[str, Any] = read_yaml_file(self._config.schema_path)

        self._target_cols: List[str] = self._schema.get("target_cols", [])
        self._text_cols: List[str] = self._schema.get("text_cols", [])
        self._drop_cols: Any = self._schema.get("drop_cols", []) or []

        if isinstance(self._drop_cols, str):
            self._drop_cols = [self._drop_cols]

    # ----- Private Helpers -----

    def _feature_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if self._drop_cols:
                df = df.drop(columns=[c for c in self._drop_cols if c in df.columns], errors="ignore")

            df = df.drop_duplicates().reset_index(drop=True)
            df = df.dropna().reset_index(drop=True)
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def _text_preprocessing(self, df: pd.DataFrame, preprocessor: CustomPreprocessor) -> pd.DataFrame:
        try:
            for col in self._text_cols:
                if col not in df.columns:
                    continue
                df[col] = preprocessor.transform(df[col].astype(str).values)

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def _split_features_targets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            X = df[self._text_cols].copy()
            y = df[self._target_cols].copy()
            return X, y

        except Exception as e:
            raise CustomException(e, sys)

    def _save_preprocessor(self, preprocessor: CustomPreprocessor) -> None:
        try:
            save_object(self._config.preprocessor_file_path, preprocessor)
        except Exception as e:
            raise CustomException(e, sys)

    # ----- Public Orchestrator -----

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            train_df = read_csv_file(train_path)
            test_df = read_csv_file(test_path)

            train_df = self._feature_cleaning(train_df)
            test_df = self._feature_cleaning(test_df)

            preprocessor = CustomPreprocessor()

            train_df = self._text_preprocessing(train_df, preprocessor)
            test_df = self._text_preprocessing(test_df, preprocessor)

            X_train_df, y_train_df = self._split_features_targets(train_df)
            X_test_df, y_test_df = self._split_features_targets(test_df)

            self._save_preprocessor(preprocessor)

            return (
                X_train_df.values,
                X_test_df.values,
                y_train_df.values,
                y_test_df.values,
            )

        except Exception as e:
            raise CustomException(e, sys)



# ---- Testing ----
if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion, DataIngestionConfig

    ingest_config = DataIngestionConfig()
    data_ingestion = DataIngestion(config=ingest_config)

    train_path, test_path = data_ingestion.initiate_data_ingestion()

    transformer = DataTransformation()

    X_train, X_test, y_train, y_test = transformer.initiate_data_transformation(
        train_path=train_path,
        test_path=test_path
    )

    print("✅ X_train:", X_train.shape)
    print("✅ X_test:", X_test.shape)
    print("✅ y_train:", y_train.shape)
    print("✅ y_test:", y_test.shape)
