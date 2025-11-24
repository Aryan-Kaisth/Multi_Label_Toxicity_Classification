import re
import sys
import contractions
import unicodedata
import emoji
import spacy
from src.logger import logging
from src.exception import CustomException


def to_lowercase(text: str) -> str:
    """
    Convert the input text to lowercase.

    Parameters
    ----------
    text : str
        Input text to convert.

    Returns
    -------
    str
        Lowercased text.

    Raises
    ------
    CustomException
        If the conversion fails.
    """
    try:
        logging.info("[TEXT] Converting text to lowercase.")
        return text.lower()

    except Exception as e:
        logging.error("[TEXT] Lowercase conversion failed.")
        raise CustomException(e, sys)


def remove_urls(text: str) -> str:
    """
    Remove URLs from the input text using a regular expression.

    Parameters
    ----------
    text : str
        Input text that may contain URLs.

    Returns
    -------
    str
        Text with all URLs removed.

    Raises
    ------
    CustomException
        If URL removal fails.
    """
    try:
        logging.info("[TEXT] Removing URLs from text.")
        return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    except Exception as e:
        logging.error("[TEXT] URL removal failed.")
        raise CustomException(e, sys)


def expand_contractions(text: str) -> str:
    """
    Expand English contractions in the input text.

    Parameters
    ----------
    text : str
        Text containing contractions.

    Returns
    -------
    str
        Text with contractions expanded.

    Raises
    ------
    CustomException
        If expansion fails.
    """
    try:
        logging.info("[TEXT] Expanding contractions.")
        return contractions.fix(text)

    except Exception as e:
        logging.error("[TEXT] Contraction expansion failed.")
        raise CustomException(e, sys)


def remove_accents_diacritics(text: str) -> str:
    """
    Remove accents and diacritical marks from the input text.

    Parameters
    ----------
    text : str
        Text that may contain accented characters.

    Returns
    -------
    str
        Normalized text with accents and diacritics removed.

    Raises
    ------
    CustomException
        If normalization fails.
    """
    try:
        logging.info("[TEXT] Removing accents and diacritics.")
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])
        return text

    except Exception as e:
        logging.error("[TEXT] Failed to remove accents/diacritics.")
        raise CustomException(e, sys)


def convert_emojis(text: str) -> str:
    """
    Convert emojis in the input text into their textual descriptions.

    Parameters
    ----------
    text : str
        Text containing emoji characters.

    Returns
    -------
    str
        Text with emojis replaced by descriptive names.

    Raises
    ------
    CustomException
        If emoji conversion fails.
    """
    try:
        logging.info("[TEXT] Converting emojis to text representations.")
        return emoji.demojize(text)

    except Exception as e:
        logging.error("[TEXT] Emoji conversion failed.")
        raise CustomException(e, sys)


def remove_mentions(text: str) -> str:
    """
    Remove @mentions from the input text and normalize spacing.

    Parameters
    ----------
    text : str
        Input text containing @mentions.

    Returns
    -------
    str
        Text with all @mentions removed.

    Raises
    ------
    CustomException
        If mention removal fails.
    """
    try:
        logging.info("[TEXT] Removing @mentions.")
        text = re.sub(r'@[A-Za-z0-9_.-]+', '', text)
        return " ".join(text.split())

    except Exception as e:
        logging.error("[TEXT] Mention removal failed.")
        raise CustomException(e, sys)


def spacy_remove_punct_numbers_pipe(text_list):
    """
    Remove punctuation and numeric tokens from text using spaCy's pipeline.

    This function processes text in batches using `nlp.pipe()` for improved
    performance. Non-alphabetic tokens are removed, and remaining tokens
    are lowercased and joined back into cleaned strings.

    Parameters
    ----------
    text_list : list of str
        List of raw text documents.

    Returns
    -------
    list of str
        Cleaned text documents with punctuation and numbers removed.

    Raises
    ------
    CustomException
        If spaCy processing fails.
    """
    try:
        logging.info("[SPACY] Removing punctuation and numbers using nlp.pipe.")

        nlp = spacy.load("en_core_web_sm")
        cleaned = []

        for doc in nlp.pipe(text_list, batch_size=500, n_process=-1):
            tokens = [token.text.lower() for token in doc if token.is_alpha]
            cleaned.append(" ".join(tokens))

        logging.info("[SPACY] Punctuation/number removal completed.")
        return cleaned

    except Exception as e:
        logging.error("[SPACY] Failed during punctuation/number removal.")
        raise CustomException(e, sys)


def spacy_lemmatize_pipe(text_list):
    """
    Lemmatize text documents using spaCy's optimized processing pipeline.

    Each token is replaced with its lemma form, and the output is returned
    as a list of lemmatized strings.

    Parameters
    ----------
    text_list : list of str
        Input text documents to lemmatize.

    Returns
    -------
    list of str
        Lemmatized text documents.

    Raises
    ------
    CustomException
        If lemmatization fails.
    """
    try:
        logging.info("[SPACY] Lemmatizing text using nlp.pipe.")

        nlp = spacy.load("en_core_web_sm")
        lemmatized = []

        for doc in nlp.pipe(text_list, batch_size=500, n_process=-1):
            lemmas = [token.lemma_ for token in doc]
            lemmatized.append(" ".join(lemmas))

        logging.info("[SPACY] Lemmatization completed.")
        return lemmatized

    except Exception as e:
        logging.error("[SPACY] Lemmatization failed.")
        raise CustomException(e, sys)


def spacy_tokenize_pipe(text_list):
    """
    Tokenize text documents using spaCy's batched pipeline.

    Parameters
    ----------
    text_list : list of str
        Input text documents.

    Returns
    -------
    list of list of str
        Tokenized documents where each document is represented as a list of tokens.

    Raises
    ------
    CustomException
        If tokenization fails.
    """
    try:
        logging.info("[SPACY] Tokenizing text using nlp.pipe.")

        nlp = spacy.load("en_core_web_sm")
        tokenized = []

        for doc in nlp.pipe(text_list, batch_size=500, n_process=-1):
            tokenized.append([token.text for token in doc])

        logging.info("[SPACY] Tokenization completed.")
        return tokenized

    except Exception as e:
        logging.error("[SPACY] Tokenization failed.")
        raise CustomException(e, sys)
