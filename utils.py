import torch
import pandas as pd

from pathlib import Path

def save_model(model: torch.nn.Module,
               saving_dir: str,
               model_name: str):
    '''
    Saves a pytorch model's state dictionary to a given path.

    Args:
        model: PyTorch trained model.
        saving_dir: Directory where you want to save your model.
        model_name: name of the model.
    '''
    saving_dir_path = Path(saving_dir)
    saving_dir_path.mkdir(parents=True,
                          exist_ok=True)

    model_name = model_name + '.pth'
    model_save_path = saving_dir_path / model_name

    print(f'[INFO] Saving model to: {model_save_path}')
    torch.save(obj=model.state_dict(), f=model_save_path)

def save_training_result(result, model_name):
    '''
    Takes a model result asa dictionary and saves it as csv.

    Args:
        result: model result dictionary.
        model_name: name of the model.
    '''
    path = model_name + "_result.csv"
    result_df = pd.DataFrame(result)
    result_df.to_csv(path, index=False)
    print(f'Training result saved successfully...')