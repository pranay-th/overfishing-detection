# Overfishing Detection Project

This project focuses on detecting overfishing activities using a combination of satellite imagery and AIS (Automatic Identification System) data. By leveraging machine learning models, such as XGBoost, LSTM, and YOLOv8, the project aims to identify fishing vessels and predict overfishing patterns.

## **Key Features**
- **Satellite Image Analysis**: Detects ships using Sentinel-1 SAR images and YOLOv8 object detection.
- **AIS Data Classification**: Classifies fishing activities using AIS data with XGBoost and LSTM models.
- **Visualization**: Provides detailed visualizations of predictions, metrics, and data trends.
- **Cross-Validation**: Implements k-fold cross-validation to evaluate model performance.


![Image](https://github.com/user-attachments/assets/47911a25-73ec-4aa8-b975-e9c2bfe1e597)


## **Project Structure**
- **data/**: Contains datasets used in the project.
  - **satellite/**: Directory for satellite imagery data.
    - **sentinel1-sar-sample-data/**: Sample Sentinel-1 SAR images.
  - **ais/**: Directory for AIS training data.
    - **ais-sample-data/**: Sample AIS data from the Global Fishing Watch API.

- **notebooks/**: Jupyter notebooks for data preprocessing, visualization, and model training.
  - **data_preprocessing.ipynb**: Preprocesses satellite and AIS data.
  - **model_training.ipynb**: Trains models for overfishing detection.
  - **visualization.ipynb**: Generates visualizations for data and model metrics.

- **src/**: Source code for the project.
  - **ais_model.py**: XGBoost model for AIS data classification.
  - **ais_lstm_model.py**: LSTM model for AIS data classification.
  - **predict_ships.py**: YOLOv8-based ship detection for satellite images.
  - **data_loader.py**: Functions to load and preprocess AIS and satellite data.

- **yolov8_ship_det_satellite/**: Directory for YOLOv8 model and predictions.
  - **ship.pt**: Pre-trained YOLOv8 model for ship detection.
  - **predictions/**: Contains annotated images with bounding boxes.

- **requirements.txt**: Lists the required Python packages for the project.

## **Technologies Used**
- **Python**: Core programming language.
- **TensorFlow/Keras**: For LSTM-based AIS classification.
- **XGBoost**: For AIS data classification.
- **YOLOv8**: For ship detection in satellite images.
- **Matplotlib/Seaborn**: For data visualization.

## **Research Objective**
This project aims to analyze the impact of overfishing by detecting fishing vessels and predicting overfishing activities. The findings will contribute to better marine resource management and conservation strategies.