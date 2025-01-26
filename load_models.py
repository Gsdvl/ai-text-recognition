import joblib

def load_models():
    with open('models/modelv2.pkl', 'rb') as model_file:
        model = joblib.load(model_file)

    # Carregar o vectorizer
    with open('models/vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = joblib.load(vectorizer_file)

    return model,vectorizer
