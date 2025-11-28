import re
import sys
import contractions
import unicodedata
import emoji
import spacy
from src.exception import CustomException


def to_lowercase(text: str) -> str:
    """
    Convert text to lowercase.

    Parameters
    ----------
    text : str
        Input text.

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
        return text.lower()
    except Exception as e:
        raise CustomException(e, sys)


def remove_urls(text: str) -> str:
    """
    Remove URLs from text.

    Parameters
    ----------
    text : str
        Input text containing URLs.

    Returns
    -------
    str
        Text with URLs removed.

    Raises
    ------
    CustomException
        If URL removal fails.
    """
    try:
        return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    except Exception as e:
        raise CustomException(e, sys)


def expand_contractions(text: str) -> str:
    """
    Expand English contractions.

    Parameters
    ----------
    text : str
        Input text containing contractions.

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
        return contractions.fix(text)
    except Exception as e:
        raise CustomException(e, sys)


def remove_accents_diacritics(text: str) -> str:
    """
    Remove accents and diacritical marks from text.

    Parameters
    ----------
    text : str
        Input text containing diacritics.

    Returns
    -------
    str
        Normalized text without diacritics.

    Raises
    ------
    CustomException
        If normalization fails.
    """
    try:
        text = unicodedata.normalize('NFKD', text)
        return ''.join([c for c in text if not unicodedata.combining(c)])
    except Exception as e:
        raise CustomException(e, sys)


def convert_emojis(text: str) -> str:
    """
    Convert emoji characters to their text descriptions.

    Parameters
    ----------
    text : str
        Input text containing emojis.

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
        return emoji.demojize(text)
    except Exception as e:
        raise CustomException(e, sys)


def remove_mentions(text: str) -> str:
    """
    Remove @mentions from text.

    Parameters
    ----------
    text : str
        Input text containing mentions.

    Returns
    -------
    str
        Text with mentions removed and spacing normalized.

    Raises
    ------
    CustomException
        If mention removal fails.
    """
    try:
        text = re.sub(r'@[A-Za-z0-9_.-]+', '', text)
        return " ".join(text.split())
    except Exception as e:
        raise CustomException(e, sys)


try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    raise CustomException(e, sys)


def spacy_remove_punct_numbers_pipe(text_list: list) -> list:
    """
    Remove punctuation and numbers using spaCy's pipeline.

    Parameters
    ----------
    text_list : list of str
        List of input documents.

    Returns
    -------
    list of str
        Texts containing only alphabetic tokens.

    Raises
    ------
    CustomException
        If processing fails.
    """
    try:
        cleaned = []
        for doc in nlp.pipe(text_list, batch_size=500, n_process=-1):
            tokens = [token.text.lower() for token in doc if token.is_alpha]
            cleaned.append(" ".join(tokens))
        return cleaned
    except Exception as e:
        raise CustomException(e, sys)


def spacy_lemmatize_pipe(text_list: list) -> list:
    """
    Lemmatize text documents using spaCy.

    Parameters
    ----------
    text_list : list of str
        List of input documents.

    Returns
    -------
    list of str
        Lemmatized text.

    Raises
    ------
    CustomException
        If lemmatization fails.
    """
    try:
        lemmatized = []
        for doc in nlp.pipe(text_list, batch_size=500, n_process=-1):
            lemmas = [token.lemma_ for token in doc]
            lemmatized.append(" ".join(lemmas))
        return lemmatized
    except Exception as e:
        raise CustomException(e, sys)


def spacy_tokenize_pipe(text_list: list) -> list:
    """
    Tokenize text using spaCy.

    Parameters
    ----------
    text_list : list of str
        List of input documents.

    Returns
    -------
    list of list of str
        Tokenized documents.

    Raises
    ------
    CustomException
        If tokenization fails.
    """
    try:
        tokenized = []
        for doc in nlp.pipe(text_list, batch_size=500, n_process=-1):
            tokenized.append([token.text for token in doc])
        return tokenized
    except Exception as e:
        raise CustomException(e, sys)
