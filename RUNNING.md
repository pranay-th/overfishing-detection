# How to Run the Overfishing Detection Project

Follow these steps to set up and run the project:

## **Setup Instructions**
1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd overfishing-detection
   ```

2. **Install Dependencies**:
   Install the required Python packages using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Data**:
   - Place Sentinel-1 SAR images in the `data/satellite/sentinel1-sar-sample-data/` directory.
   - Place AIS training data in the `data/ais/ais-sample-data/` directory.

4. **Set Up YOLOv8**:
   - Ensure the YOLOv8 model (`ship.pt`) is located in the `yolov8_ship_det_satellite/` directory.

## **Running the Project**

### **1. Preprocess the Data**
Run the `data_preprocessing.ipynb` notebook to preprocess the satellite and AIS data:
```bash
jupyter notebook notebooks/data_preprocessing.ipynb
```

### **2. Train the Models**
- **AIS Data Classification**:
  - Run `ais_model.py` for XGBoost-based classification:
    ```bash
    python src/ais_model.py
    ```
  - Run `ais_lstm_model.py` for LSTM-based classification:
    ```bash
    python src/ais_lstm_model.py
    ```

- **Satellite Image Ship Detection**:
  - Run `predict_ships.py` to detect ships in satellite images:
    ```bash
    python yolov8_ship_det_satellite/predict_ships.py
    ```

### **3. Visualize Results**
- Open `visualization.ipynb` to generate visualizations for data and model metrics:
  ```bash
  jupyter notebook notebooks/visualization.ipynb
  ```

## **Outputs**
- **AIS Model Metrics**:
  - ROC curves and metrics improvement plots are saved in the `src/` directory.
- **YOLOv8 Predictions**:
  - Annotated satellite images with bounding boxes are saved in `yolov8_ship_det_satellite/predictions/`.

## **Troubleshooting**
- Ensure all dependencies are installed using the correct Python environment.
- Verify that the data is placed in the correct directories.
- For YOLOv8 issues, ensure the `ultralytics` package is installed:
  ```bash
  pip install ultralytics
  ```

Let me know if you encounter any issues!