# COVID Face Mask Detection

This project implements a Convolutional Neural Network (CNN) to detect whether a person is wearing a face mask in images and videos. The model was developed as part of an MSC CS final year project in 2022 and is now reimplemented for a Git repository.

## Dataset
- Source: [COVID Face Mask Detection Dataset](https://www.kaggle.com/datasets/prithwirajmitra/covid-face-mask-detection-dataset)
- License: Copyright Authors
- Structure: Images are divided into `Train`, `Test`, and `Validation` sets with `Mask` and `Non Mask` classes.

## Project Structure
Covid Face Mask Detection
- ├── Sample Pictures\                # Sample images and video for testing
- ├── models\                         # Trained model file
- ├── notebooks\                      # Jupyter notebook for training
- ├── scripts\                        # Python scripts for training and detection
- ├── haarcascade_frontalface_default.xml
- ├── requirements.txt
- ├── README.md
- ├── LICENSE
- ├── .gitignore
- └── data_download.py


## Setup
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd Covid-Face-Mask-Detection

2. Create a virtual environment and install dependencies:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    pip install -r requirements.txt
    
3. Download and extract the dataset:
    ```bash
   python data_download.py
   
4. Train the model:
    - Open notebooks/face_mask_detection.ipynb in Jupyter Notebook and run all cells.
    - Alternatively, run:
    ```bash
   python scripts/train.py
   
5. Test the model on sample images or video:
    - For static images, run the prediction section in face_mask_detection.ipynb.
    - For video, run:
    ```bash
   python scripts/mask_detect.py

## Requirements
See requirements.txt for a list of dependencies.

## Usage
- Training: The notebook or train.py trains a CNN model and saves it to models/model.h5.
- Static Image Prediction: The notebook predicts mask presence in images in Sample Pictures.
- Video Detection: mask_detect.py uses Haar Cascade for face detection and the CNN model to classify masks in Sample Pictures/video.mp4, saving face images to input.

## License
- This project is licensed under the MIT License. See the LICENSE file for details.
