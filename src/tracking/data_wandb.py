import wandb
import os

import params

RAW_PATH = 'data/raw/'
INTER_PATH = 'data/inter/'

def log_data(data_dir: str, run, artifact_name: str, artifact_type: str="data") -> None:
    artifact = wandb.Artifact(name=artifact_name, type=artifact_type)

    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        if os.path.isfile(file_path):
            artifact.add_file(file_path)

    run.log_artifact(artifact)

if __name__ == "__main__":
    run = wandb.init(project=params.WANDB_PROJECT, entity=params.ENTITY)
    log_data(RAW_PATH, run, "raw_data", "raw")
