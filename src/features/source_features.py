"""
File: source_features.py

Purpose
-------
Source credibility feature engineering module for TruthLens AI.

This module extracts domain-level credibility signals from news sources
to help detect misinformation originating from unreliable outlets.

Input
-----
df : pandas.DataFrame

Required columns:
    source : str (URL or domain)

Output
------
pandas.DataFrame
    Original dataframe augmented with source credibility features.
"""

import logging
from urllib.parse import urlparse
from typing import Optional

import pandas as pd

from src.utils.input_validation import ensure_dataframe


# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Known Source Lists
# ---------------------------------------------------------

HIGH_CREDIBILITY_SOURCES = {
    "bbc.com",
    "reuters.com",
    "nytimes.com",
    "theguardian.com",
    "apnews.com",
    "wsj.com",
}

LOW_CREDIBILITY_SOURCES = {
    "infowars.com",
    "beforeitsnews.com",
    "naturalnews.com",
}


# ---------------------------------------------------------
# Domain Extraction
# ---------------------------------------------------------

def _extract_domain(url: Optional[str]) -> str:
    """
    Extract domain name from a URL.

    Parameters
    ----------
    url : str

    Returns
    -------
    str
        Cleaned domain name.
    """

    try:

        if url is None or pd.isna(url):
            return ""

        parsed = urlparse(str(url))

        domain = parsed.netloc.lower()

        if domain.startswith("www."):
            domain = domain[4:]

        return domain

    except Exception:

        return ""


# ---------------------------------------------------------
# Credibility Scoring
# ---------------------------------------------------------

def _source_credibility(domain: str) -> int:
    """
    Assign credibility score to domain.

    Returns
    -------
    int
        1  = high credibility
        0  = unknown
        -1 = low credibility
    """

    if domain in HIGH_CREDIBILITY_SOURCES:
        return 1

    if domain in LOW_CREDIBILITY_SOURCES:
        return -1

    return 0


# ---------------------------------------------------------
# Feature Engineering Pipeline
# ---------------------------------------------------------

def add_source_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add source credibility features to dataset.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Dataset with additional source credibility features.
    """

    try:

        ensure_dataframe(df, name="df")

        logger.info("Starting source credibility feature extraction")

        if "source" not in df.columns:

            logger.warning("Column 'source' not found. Skipping source features.")

            return df

        # -------------------------------------------------
        # Domain Extraction
        # -------------------------------------------------

        df["domain"] = df["source"].apply(_extract_domain)

        # -------------------------------------------------
        # Credibility Score
        # -------------------------------------------------

        df["source_credibility"] = df["domain"].apply(_source_credibility)

        # -------------------------------------------------
        # Binary Credibility Signals
        # -------------------------------------------------

        df["is_high_credibility"] = (df["source_credibility"] == 1).astype(int)

        df["is_low_credibility"] = (df["source_credibility"] == -1).astype(int)

        # -------------------------------------------------
        # Domain Frequency
        # -------------------------------------------------

        domain_counts = df["domain"].value_counts()

        df["domain_frequency"] = df["domain"].map(domain_counts)

        logger.info("Source credibility feature extraction completed")

        return df

    except Exception:

        logger.exception("Source credibility feature extraction failed")

        raise


# ---------------------------------------------------------
# Backward Compatibility Wrapper
# ---------------------------------------------------------

def add_source_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible wrapper for legacy code.
    """

    return add_source_features(df)