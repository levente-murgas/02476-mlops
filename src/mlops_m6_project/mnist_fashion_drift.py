import numpy as np
import pandas as pd
from torchvision import datasets
from PIL import Image
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from tqdm import tqdm


def calculate_brightness(image: np.ndarray) -> float:
    """Calculate average brightness of an image."""
    return float(np.mean(image))


def calculate_contrast(image: np.ndarray) -> float:
    """Calculate contrast (standard deviation) of an image."""
    return float(np.std(image))


def calculate_sharpness(image: np.ndarray) -> float:
    """Calculate sharpness using Laplacian variance."""
    # Compute gradients
    dx = np.diff(image, axis=1)
    dy = np.diff(image, axis=0)
    # Calculate variance of gradients as a measure of sharpness
    sharpness = float(np.var(dx) + np.var(dy))
    return sharpness


def extract_features(dataset) -> pd.DataFrame:
    """Extract brightness, contrast, and sharpness features from dataset."""
    features = []
    
    for idx in tqdm(range(len(dataset)), desc="Extracting features"):
        image, label = dataset[idx]
        # Convert to numpy array
        if hasattr(image, 'numpy'):
            img_array = image.numpy().squeeze()
        else:
            img_array = np.array(image)
        
        # Calculate features
        brightness = calculate_brightness(img_array)
        contrast = calculate_contrast(img_array)
        sharpness = calculate_sharpness(img_array)
        
        features.append({
            'brightness': brightness,
            'contrast': contrast,
            'sharpness': sharpness,
            'label': label
        })
    
    return pd.DataFrame(features)


def main():
    print("Downloading MNIST dataset...")
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True)
    
    print("Downloading FashionMNIST dataset...")
    fashion_dataset = datasets.FashionMNIST(root='./data', train=True, download=True)
    
    print("\nExtracting features from MNIST...")
    mnist_features = extract_features(mnist_dataset)
    
    print("\nExtracting features from FashionMNIST...")
    fashion_features = extract_features(fashion_dataset)
    
    # Display statistics
    print("\n" + "="*60)
    print("MNIST Statistics:")
    print("="*60)
    print(mnist_features[['brightness', 'contrast', 'sharpness']].describe())
    
    print("\n" + "="*60)
    print("FashionMNIST Statistics:")
    print("="*60)
    print(fashion_features[['brightness', 'contrast', 'sharpness']].describe())
    
    # Run Evidently data drift report
    print("\n" + "="*60)
    print("Running Evidently Data Drift Analysis...")
    print("="*60)
    
    # Use MNIST as reference and FashionMNIST as current
    report = Report(metrics=[
        DataSummaryPreset(),
        DataDriftPreset(),
    ])
    
    # Drop label column for drift analysis
    mnist_features_no_label = mnist_features[['brightness', 'contrast', 'sharpness']]
    fashion_features_no_label = fashion_features[['brightness', 'contrast', 'sharpness']]
    
    snapshot = report.run(
        reference_data=mnist_features_no_label,
        current_data=fashion_features_no_label
    )
    
    # Save report
    output_path = 'mnist_fashion_drift_report.html'
    snapshot.save_html(output_path)
    print(f"\nDrift report saved to: {output_path}")
    
    # Save feature dataframes
    mnist_features.to_csv('mnist_features.csv', index=False)
    fashion_features.to_csv('fashion_features.csv', index=False)
    print("Feature dataframes saved to mnist_features.csv and fashion_features.csv")


if __name__ == "__main__":
    main()
