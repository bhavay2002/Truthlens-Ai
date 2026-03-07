from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_features(texts):

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    X = vectorizer.fit_transform(texts)

    return X, vectorizer
