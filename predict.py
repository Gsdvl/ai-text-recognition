from load_models import load_models

def predict(text):
    model, vectorizer = load_models()

    X = vectorizer.transform([text])

    if hasattr(X, "toarray"):  # Verifica se o método 'toarray' exist
        X = X.toarray()

    prediction = model.predict(X)
    return prediction
