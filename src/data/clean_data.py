import pandas as pd
import re
import logging
import unicodedata

logger = logging.getLogger(__name__)


# -------------------------------------------------
# Precompiled Regex (faster)
# -------------------------------------------------

URL_PATTERN = re.compile(r"http\S+|www\S+|https\S+")
EMAIL_PATTERN = re.compile(r"\S+@\S+")
MENTION_PATTERN = re.compile(r"@\w+|#\w+")
HTML_PATTERN = re.compile(r"<.*?>")
WHITESPACE_PATTERN = re.compile(r"\s+")
NUMBER_PATTERN = re.compile(r"\d+")
REPEATED_CHARS = re.compile(r"(.)\1{2,}")

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "]+",
    flags=re.UNICODE
)


# -------------------------------------------------
# Text Normalization
# -------------------------------------------------

def normalize_unicode(text):
    """Normalize unicode characters"""
    text = unicodedata.normalize("NFKD", text)

    replacements = {
        "“": '"',
        "”": '"',
        "’": "'",
        "‘": "'",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text


def expand_contractions(text):

    try:
        import contractions
        return contractions.fix(text)

    except ImportError:
        return text


def normalize_numbers(text):

    return NUMBER_PATTERN.sub("<NUM>", text)


def remove_emojis(text):

    return EMOJI_PATTERN.sub("", text)


def normalize_repeated_chars(text):

    return REPEATED_CHARS.sub(r"\1\1", text)


# -------------------------------------------------
# Core Cleaning
# -------------------------------------------------

def clean_text(text, normalize_nums=True):

    original_text = text  # Keep original for fallback
    text = str(text)

    text = normalize_unicode(text)

    text = remove_emojis(text)

    text = text.lower()

    text = expand_contractions(text)

    text = URL_PATTERN.sub("", text)

    text = EMAIL_PATTERN.sub("", text)

    text = MENTION_PATTERN.sub("", text)

    text = HTML_PATTERN.sub("", text)

    text = normalize_repeated_chars(text)

    if normalize_nums:
        text = normalize_numbers(text)

    text = re.sub(r"[!?]{2,}", "!", text)
    text = re.sub(r"[.]{2,}", ".", text)

    text = re.sub(r"[^a-zA-Z0-9\s.,!?<>]", "", text)

    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    # Fallback: if cleaning removed everything, return original lowercased
    if not text or len(text) < 3:
        return str(original_text).lower().strip()

    return text


# -------------------------------------------------
# DataFrame Cleaning
# -------------------------------------------------

def clean_dataframe(
    df,
    text_column="text",
    title_column=None,
    min_len=30
):

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found")

    df = df.copy()

    initial_rows = len(df)

    logger.info(f"Initial dataset size: {initial_rows}")

    # Merge title + text if available
    if title_column and title_column in df.columns:

        df[text_column] = (
            df[title_column].fillna("") +
            " </s> " +
            df[text_column].fillna("")
        )

    # Remove duplicates
    df = df.drop_duplicates(subset=[text_column])

    # Remove missing
    df = df.dropna(subset=[text_column])

    df[text_column] = df[text_column].astype(str).str.strip()

    # Remove empty
    df = df[df[text_column].str.len() > 0]

    # Apply cleaning
    df[text_column] = df[text_column].apply(clean_text)

    # Remove short texts
    df["word_count"] = df[text_column].apply(
        lambda x: len(x.split())
    )

    df = df[df["word_count"] >= min_len]

    df = df.drop(columns=["word_count"])

    df = df.reset_index(drop=True)

    final_rows = len(df)

    logger.info(f"Final dataset size: {final_rows}")
    logger.info(f"Rows removed: {initial_rows - final_rows}")

    retention = (final_rows / initial_rows) * 100 if initial_rows else 0

    logger.info(f"Retention rate: {retention:.2f}%")

    return df


# -------------------------------------------------
# Optional NLP preprocessing
# -------------------------------------------------

def advanced_text_preprocessing(
    text,
    remove_stopwords=False,
    lemmatize=False
):

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

        logger.warning(
            "NLTK not installed, skipping advanced preprocessing"
        )

    return text
