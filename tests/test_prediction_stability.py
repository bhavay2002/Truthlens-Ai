class DummyModel:

    def predict(self, text):
        return {"label": "REAL", "confidence": 0.9}


def test_prediction_stability():

    model = DummyModel()

    text = "Breaking news: new technology released."

    pred1 = model.predict(text)
    pred2 = model.predict(text)

    assert pred1 == pred2