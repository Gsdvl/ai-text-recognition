from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import argparse, os
import numpy as np
import pandas as pd

def get_data():
    #TODO CRIAR A FUNÇÃO QUE BAIXA OS DADOS DO WANDB
    data = pd.read_csv('../results/balanced.csv')
    return data

def vectorize_data(data):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(data).toarray()
    return X_tfidf

def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(5000,)),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid') 
    ])

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    model.summary()
    return model


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
    run = wandb.init(project=params.WANDB_PROJECT, entity=params.ENTITY, job_type="training")
    train_generator, validation_generator = generate_train()
    model = create_model()
    train_model(model, train_generator, validation_generator)
    run.link_model(path="model.keras", registered_model_name="flower_classifier_besthparams")
    run.finish()
    wandb.finish()


compile_model()