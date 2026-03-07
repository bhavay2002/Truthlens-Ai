from lime.lime_text import LimeTextExplainer


def explain_prediction(predict_fn, text):

    explainer = LimeTextExplainer(class_names=["Real", "Fake"])

    exp = explainer.explain_instance(
        text,
        predict_fn,
        num_features=10
    )

    exp.show_in_notebook()
