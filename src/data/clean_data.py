import pandas as pd
import re
import logging
import unicodedata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------------------------
# Text Normalization
# -------------------------------------------------

def normalize_unicode(text):
    """
    Normalize unicode characters
    Example: fancy quotes → standard quotes
    """
    return unicodedata.normalize("NFKD", text)


def expand_contractions(text):
    """
    Expand contractions like don't → do not
    """
    try:
        import contractions
        return contractions.fix(text)
    except ImportError:
        return text


def normalize_numbers(text):
    """
    Replace numbers with token <NUM>
    """
    return re.sub(r"\d+", "<NUM>", text)


def remove_emojis(text):
    """
    Remove emojis and other unicode symbols
    """
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


# -------------------------------------------------
# Core Text Cleaning
# -------------------------------------------------

def clean_text(text, normalize_nums=True):
    """
    Professional text cleaning pipeline
    """

    text = str(text)

    # Unicode normalization
    text = normalize_unicode(text)

    # Remove emojis
    text = remove_emojis(text)

    # Lowercase
    text = text.lower()

    # Expand contractions
    text = expand_contractions(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove emails
    text = re.sub(r"\S+@\S+", "", text)

    # Remove mentions / hashtags
    text = re.sub(r"@\w+|#\w+", "", text)

    # Remove HTML
    text = re.sub(r"<.*?>", "", text)

    # Normalize numbers
    if normalize_nums:
        text = normalize_numbers(text)

    # Normalize repeated punctuation
    text = re.sub(r"[!?]{2,}", "!", text)
    text = re.sub(r"[.]{2,}", ".", text)

    # Remove unwanted characters
    text = re.sub(r"[^a-zA-Z0-9\s.,!?<>]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -------------------------------------------------
# DataFrame Cleaning
# -------------------------------------------------

def clean_dataframe(df, text_column="text", min_len=10):
    """
    Professional dataset cleaning
    """

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in dataframe")

    df = df.copy()
    initial_rows = len(df)
    logger.info(f"Initial dataset size: {initial_rows}")

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove missing
    df = df.dropna(subset=[text_column])

    # Strip whitespace
    df[text_column] = df[text_column].astype(str).str.strip()

    # Remove empty rows
    df = df[df[text_column].str.len() > 0]

    # Remove short rows
    df = df[df[text_column].str.len() >= min_len]

    # Apply text cleaning
    df[text_column] = df[text_column].apply(clean_text)

    # Remove rows empty after cleaning
    df = df[df[text_column].str.len() >= min_len]

    df = df.reset_index(drop=True)

    final_rows = len(df)

    logger.info(f"Final dataset size: {final_rows}")
    logger.info(f"Rows removed: {initial_rows - final_rows}")
    retention = (final_rows / initial_rows) * 100 if initial_rows else 0.0
    logger.info(f"Retention rate: {retention:.2f}%")

    return df


# -------------------------------------------------
# Advanced NLP Preprocessing
# -------------------------------------------------

def advanced_text_preprocessing(
    text,
    remove_stopwords=False,
    lemmatize=False
):
    """
    Optional advanced preprocessing
    """

    text = clean_text(text)

    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)

        words = text.split()

        if remove_stopwords:
            stop_words = set(stopwords.words("english"))
            words = [w for w in words if w not in stop_words]

        if lemmatize:
            lemmatizer = WordNetLemmatizer()
            words = [lemmatizer.lemmatize(w) for w in words]

        text = " ".join(words)

    except ImportError:
        logger.warning("NLTK not installed, skipping advanced preprocessing")

    return text
