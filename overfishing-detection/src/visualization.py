import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def plot_ais_data(ais_data):
    plt.figure(figsize=(12, 6))
    sns.countplot(data=ais_data, x='fishing_activity', palette='viridis')
    plt.title('Fishing Activity Distribution')
    plt.xlabel('Fishing Activity')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_sar_images(sar_images):
    fig, axes = plt.subplots(1, len(sar_images), figsize=(15, 5))
    for ax, img in zip(axes, sar_images):
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.suptitle('Sample Sentinel 1 SAR Images')
    plt.tight_layout()
    plt.show()

def plot_overfishing_trends(data):
    plt.figure(figsize=(12, 6))
    data.groupby('date')['fishing_activity'].sum().plot()
    plt.title('Overfishing Trends Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Fishing Activity')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()