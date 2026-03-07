"""
Source Credibility Feature Engineering
Used to estimate reliability of news sources
"""

import pandas as pd
import logging
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Known Source Lists
# -------------------------------------------------

HIGH_CREDIBILITY_SOURCES = {
    "bbc.com",
    "reuters.com",
    "nytimes.com",
    "theguardian.com",
    "apnews.com",
    "wsj.com"
}

LOW_CREDIBILITY_SOURCES = {
    "infowars.com",
    "beforeitsnews.com",
    "naturalnews.com"
}


# -------------------------------------------------
# Domain Extraction
# -------------------------------------------------

def extract_domain(url):
    """
    Extract domain name from URL
    """

    try:
        parsed = urlparse(str(url))
        domain = parsed.netloc.lower()

        if domain.startswith("www."):
            domain = domain[4:]

        return domain

    except Exception:
        return ""


# -------------------------------------------------
# Source Credibility Score
# -------------------------------------------------

def source_credibility(domain):
    """
    Assign credibility score to domain

    1  = high credibility
    0  = unknown
    -1 = low credibility
    """

    if domain in HIGH_CREDIBILITY_SOURCES:
        return 1

    if domain in LOW_CREDIBILITY_SOURCES:
        return -1

    return 0


# -------------------------------------------------
# Feature Engineering
# -------------------------------------------------

def add_source_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add source credibility features to dataframe
    """

    try:

        logger.info("Extracting source credibility features...")

        if "source" not in df.columns:
            logger.warning("No 'source' column found. Skipping source features.")
            return df

        # Extract domain
        df["domain"] = df["source"].apply(extract_domain)

        # Credibility score
        df["source_credibility"] = df["domain"].apply(source_credibility)

        # Binary high credibility feature
        df["is_high_credibility"] = (df["source_credibility"] == 1).astype(int)

        # Binary low credibility feature
        df["is_low_credibility"] = (df["source_credibility"] == -1).astype(int)

        # Domain frequency feature
        domain_counts = df["domain"].value_counts()
        df["domain_frequency"] = df["domain"].map(domain_counts)

        logger.info("Source feature extraction completed")

        return df

    except Exception as e:

        logger.error(f"Source feature extraction failed: {e}")
        raise


def add_source_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper."""
    return add_source_features(df)
