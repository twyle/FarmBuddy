import torch


def load_model(model_path: str, device: str = 'cpu'):
    return torch.load(model_path, map_location=device)

MAIZE_LABELS: list[str] = [
    'Maize Leaf Rust',
    'Northern Leaf Blight',
    'Healthy',
    'Gray Leaf Spot'
]
PEST_LABELS: list[str] = [
    'Ant',
    'Bee',
    'Beetle',
    'Catterpillar',
    'Earthworm',
    'Earwig',
    'Grasshopeer',
    'Moth',
    'Slug',
    'Snail',
    'Wasp',
    'Weevil'
]
TOMATO_LABELS: list[str] = [
    'Bacterial Spot',
    'Early Blight',
    'Late Blight',
    'Leaf Mold',
    'Septoria Leaf Spot',
    'Two Spotted Spider Mite',
    'Target Spot',
    'Yellow Leaf Curl Virus',
    'Tomato Mosaic virus',
    'Healthy',
]