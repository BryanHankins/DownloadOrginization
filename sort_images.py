import os
import shutil
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNet18_Weights  # Add this import

# Define paths
DOWNLOADS_IMAGES_FOLDER = os.path.expanduser('~/Downloads/Images')
CATEGORIES = ['Cute', 'Boring', 'NSFW', 'Others']

# Create folders for each category if they don't exist
for category in CATEGORIES:
    target_folder = os.path.join(DOWNLOADS_IMAGES_FOLDER, category)
    os.makedirs(target_folder, exist_ok=True)

# Load a pre-trained ResNet model and fine-tune it
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # Correct the variable name to 'model'

# Replace the final fully connected layer with one that has the number of categories
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(CATEGORIES))

# Load the model into evaluation mode
model.eval()

# Define transformation for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to the size expected by ResNet
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to classify images using the ResNet model
def classify_image(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output.data, 1)
    return CATEGORIES[predicted.item()]

# Function to sort images
def sort_images():
    for filename in os.listdir(DOWNLOADS_IMAGES_FOLDER):
        file_path = os.path.join(DOWNLOADS_IMAGES_FOLDER, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
            try:
                image = Image.open(file_path).convert('RGB')
                category = classify_image(image)
                target_folder = os.path.join(DOWNLOADS_IMAGES_FOLDER, category)
                shutil.move(file_path, os.path.join(target_folder, filename))
                print(f'Moved {filename} to {category} folder.')
            except Exception as e:
                print(f'Error processing {filename}: {e}')

# Execute sorting
sort_images()
print("Images sorted successfully.")
