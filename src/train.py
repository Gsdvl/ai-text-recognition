from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
import joblib
from types import SimpleNamespace
import argparse, os
import pandas as pd
import wandb

import params


def get_data():
    root = 'data/preprocessed/balanced.csv'

    if not os.path.exists(root):
        artifact = wandb.use_artifact('pedro_miguel-universidade-federal-do-rio-grande-do-norte/llm-detect/balanced_data:v0', type='dataset')
        artifact.download(root=root)
    data = pd.read_csv(root)

    return data

def vectorize_data(data):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(data).toarray()
    return X_tfidf

def create_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(5000,)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid') 
    ])

    my_optimizer = optimizers.Adam(learning_rate=model_configs.lr, beta_1=model_configs.beta_1, beta_2=model_configs.beta_2)

    model.compile(optimizer=my_optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy'])
    

    model.summary()
    return model

def train_model():
    df = get_data()

    X_Tfidf = vectorize_data(df['clean_text'])
    X_train, X_test, y_train, y_test = train_test_split(X_Tfidf, df['label'], test_size=0.2, random_state=42)
    model = create_model()

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=model_configs.epochs, batch_size=model_configs.batch_size)
    joblib.dump(model, 'models/model_by_script.pkl')

model_configs = SimpleNamespace(
    lr=0.001,
    batch_size=32,
    epochs=10,
    beta_1 = 0.9,
    beta_2 = 0.999
)
    
def parse_args():
    argparser = argparse.ArgumentParser(description="Process Hyperparameters for Model Training")
    
    argparser.add_argument('--batch_size', type=int, default=model_configs.batch_size, help='Batch size for training')
    argparser.add_argument('--lr', type=float, default=model_configs.lr, help='Learning rate for optimizer')
    argparser.add_argument('--epochs', type=int, default=model_configs.epochs, help='Number of training epochs')
    argparser.add_argument('--beta_1', type=float, default=model_configs.beta_1, help='Beta_1 parameter for Adam optimizer')
    argparser.add_argument('--beta_2', type=float, default=model_configs.beta_2, help='Beta_2 parameter for Adam optimizer')

    args = argparser.parse_args()

    model_configs.batch_size = args.batch_size
    model_configs.lr = args.lr
    model_configs.epochs = args.epochs
    model_configs.beta_1 = args.beta_1
    model_configs.beta_2 = args.beta_2

    return


if __name__ == "__main__":
    parse_args()
    run = wandb.init(project=params.WANDB_PROJECT, job_type="training")
    train_model()
    run.link_model(path="models/model_by_script.pkl", registered_model_name="model_by_script")
    run.finish()
    wandb.finish()
