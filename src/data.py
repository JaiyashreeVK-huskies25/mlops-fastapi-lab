from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def load_data():
    """Load the Wine dataset."""
    data = load_wine()
    return data.data, data.target


def split_data(X, y):
    """80/20 stratified split with a fixed seed."""
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
