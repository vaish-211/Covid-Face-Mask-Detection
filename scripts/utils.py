import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_and_extract_dataset():
    os.environ['KAGGLE_USERNAME'] = 'vaishnavi1234'    # Your Kaggle Username
    os.environ['KAGGLE_KEY'] = '925a436d659da22fc5e68794y0689hgdih'    # Your Kaggle API Key
    dataset_zip = os.path.join('..', 'covid-face-mask-detection-dataset.zip')
    dataset_dir = os.path.join('..', 'New Masks Dataset')
    if not os.path.exists(dataset_dir):
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files('prithwirajmitra/covid-face-mask-detection-dataset', path='..', unzip=False)
        os.makedirs(dataset_dir, exist_ok=True)
        with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        print(f"Dataset extracted to {dataset_dir}")