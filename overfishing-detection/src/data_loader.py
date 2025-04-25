import os
import pandas as pd
import rasterio
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def load_satellite_images(data_dir):
    """
    Load Sentinel-1 SAR images from the specified directory.
    Args:
        data_dir (str): Path to the directory containing satellite images.
    Returns:
        list: A list of numpy arrays representing the images.
    """
    images = []
    for file in os.listdir(data_dir):
        if file.endswith('.tif'):  # Sentinel-1 SAR images are often in .tif format
            file_path = os.path.join(data_dir, file)
            with rasterio.open(file_path) as src:
                images.append(src.read(1))  # Read the first band
    return images

def load_and_preprocess_ais_data(data_dir):
    """
    Load and preprocess AIS data from multiple CSV files in the specified directory.
    Args:
        data_dir (str): Path to the directory containing AIS data files.
    Returns:
        pd.DataFrame: A pandas DataFrame containing the cleaned and preprocessed AIS data.
    """
    all_data = []
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):  # Ensure only CSV files are processed
            file_path = os.path.join(data_dir, file)
            gear_type = os.path.splitext(file)[0]  # Extract gear type from filename
            df = pd.read_csv(file_path)
            df['gear_type'] = gear_type  # Add a column for gear type
            all_data.append(df)
    
    # Combine all data into a single DataFrame
    data = pd.concat(all_data, ignore_index=True)

    # Handle missing or invalid values
    data = data[data['is_fishing'] != -1]  # Remove rows with invalid 'is_fishing' values
    data = data.dropna()  # Drop rows with missing values

    # Normalize numerical features
    scaler = MinMaxScaler()
    numerical_features = ['distance_from_shore', 'distance_from_port', 'speed', 'course', 'lat', 'lon']
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    # Encode categorical features
    encoder = LabelEncoder()
    data['gear_type'] = encoder.fit_transform(data['gear_type'])

    # Ensure 'is_fishing' is binary (0 or 1)
    data['is_fishing'] = (data['is_fishing'] > 0).astype(int)

    return data

if __name__ == "__main__":
    # Example usage
    satellite_data_dir = "../data/satellite/sentinel1-sar-sample-data"
    ais_data_dir = "../data/ais/ais-sample-data"

    # Load satellite images
    satellite_images = load_satellite_images(satellite_data_dir)
    print(f"Loaded {len(satellite_images)} satellite images.")

    # Load and preprocess AIS data
    ais_data = load_and_preprocess_ais_data(ais_data_dir)
    print(f"Loaded AIS data with {ais_data.shape[0]} rows and {ais_data.shape[1]} columns.")
    print(f"Sample data:\n{ais_data.head()}")