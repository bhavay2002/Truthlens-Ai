def source_credibility(domain):

    high_cred = ["bbc.com", "reuters.com", "nytimes.com"]

    if domain in high_cred:
        return 1
    else:
        return 0


def add_source_feature(df):

    if "source" in df.columns:
        df["source_score"] = df["source"].apply(source_credibility)

    return df
