import pandas as pd
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_text(text):
    """
    Enhanced text cleaning with multiple preprocessing steps
    """
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove mentions and hashtags (social media artifacts)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters but keep spaces and basic punctuation
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def clean_dataframe(df):
    """
    Enhanced dataframe cleaning with additional checks
    """
    initial_len = len(df)
    logger.info(f"Starting with {initial_len} rows")
    
    # Remove duplicates
    df = df.drop_duplicates()
    logger.info(f"Removed {initial_len - len(df)} duplicates")
    
    # Remove rows with missing text
    df = df.dropna(subset=["text"])
    
    # Remove rows with empty text after stripping
    df = df[df['text'].astype(str).str.strip().str.len() > 0]
    
    # Remove very short texts (less than 10 characters)
    df = df[df['text'].astype(str).str.len() >= 10]
    logger.info(f"Removed short/empty texts")

    # Clean text
    df["text"] = df["text"].apply(clean_text)
    
    # Remove rows that became empty after cleaning
    df = df[df["text"].str.len() >= 10]
    
    # Reset index
    df = df.reset_index(drop=True)
    
    logger.info(f"Final dataset: {len(df)} rows ({len(df)/initial_len*100:.1f}% retained)")
    
    return df


def advanced_text_preprocessing(text, remove_stopwords=False, lemmatize=False):
    """
    Advanced preprocessing with optional stopword removal and lemmatization
    Requires: pip install nltk
    
    Args:
        text: Input text to process
        remove_stopwords: Remove common English stopwords
        lemmatize: Apply lemmatization to words
    
    Returns:
        Processed text string
    """
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        
        # Download required resources (first time only)
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
        
        text = clean_text(text)
        
        # Remove stopwords
        if remove_stopwords:
            stop_words = set(stopwords.words('english'))
            words = text.split()
            words = [w for w in words if w.lower() not in stop_words]
            text = ' '.join(words)
        
        # Lemmatization
        if lemmatize:
            lemmatizer = WordNetLemmatizer()
            words = text.split()
            words = [lemmatizer.lemmatize(w) for w in words]
            text = ' '.join(words)
        
        return text
    
    except ImportError:
        logger.warning("NLTK not available, falling back to basic cleaning")
        return clean_text(text)
