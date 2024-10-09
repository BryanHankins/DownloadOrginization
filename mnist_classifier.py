import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Improved data augmentation for CIFAR-10 and CIFAR-100
transform_cifar = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(10),  # Random rotation for data augmentation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Adjust brightness, contrast, saturation
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# MNIST transformation (remains the same)
transform_common = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize MNIST to 32x32
    transforms.Grayscale(num_output_channels=3),  # Convert MNIST to 3 channels
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_common)
cifar10_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
cifar100_trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_cifar)

trainloader_mnist = torch.utils.data.DataLoader(mnist_trainset, batch_size=64, shuffle=True)
trainloader_cifar10 = torch.utils.data.DataLoader(cifar10_trainset, batch_size=64, shuffle=True)
trainloader_cifar100 = torch.utils.data.DataLoader(cifar100_trainset, batch_size=64, shuffle=True)

# Improved neural network with additional layers, batch normalization, dropout, and better initialization
class UnifiedNet(nn.Module):
    def __init__(self):
        super(UnifiedNet, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),  # Batch normalization
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),  # Batch normalization
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Added convolutional layer
            nn.BatchNorm2d(256),  # Batch normalization
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(256 * 4 * 4, 512)  # Adjusted input size to account for the new layer
        self.dropout = nn.Dropout(0.5)  # Dropout to prevent overfitting
        # Separate output heads
        self.fc_mnist = nn.Linear(512, 10)
        self.fc_cifar10 = nn.Linear(512, 10)
        self.fc_cifar100 = nn.Linear(512, 100)
    
    def forward(self, x, dataset='mnist'):
        x = self.shared_layers(x)
        x = x.view(-1, 256 * 4 * 4)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        
        if dataset == 'mnist':
            x = self.fc_mnist(x)
        elif dataset == 'cifar10':
            x = self.fc_cifar10(x)
        elif dataset == 'cifar100':
            x = self.fc_cifar100(x)
        
        return x

# Weight initialization function
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)  # He initialization
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)  # Xavier initialization

# Create the network and apply weight initialization
net = UnifiedNet()  # Create a new instance of the network
net.apply(initialize_weights)

# Loss function and optimizer (with L2 regularization)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)  # Adam optimizer with weight decay

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop (without gradient clipping)
datasets = [('mnist', trainloader_mnist), ('cifar10', trainloader_cifar10), ('cifar100', trainloader_cifar100)]

for epoch in range(5):  # Train for more epochs to let the model converge
    for dataset_name, trainloader in datasets:
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()  # Zero gradients
            outputs = net(inputs, dataset=dataset_name)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters
            
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f'Epoch {epoch + 1}, Dataset: {dataset_name}, Batch {i + 1}, Loss: {running_loss / 100:.3f}')
                running_loss = 0.0
    
    scheduler.step()  # Step the learning rate scheduler

# Apply dynamic quantization to reduce model size
quantized_net = torch.quantization.quantize_dynamic(
    net, {torch.nn.Linear}, dtype=torch.qint8  # Only quantize Linear layers
)

# Save the quantized model's state dictionary
torch.save(quantized_net.state_dict(), 'quantized_unified_net.pth', _use_new_zipfile_serialization=True)

# Load the quantized model for inference
quantized_net = UnifiedNet()  # Create a new instance of the network
quantized_net.load_state_dict(torch.load('quantized_unified_net.pth', weights_only=True))
quantized_net.eval()  # Set the model to evaluation mode

# Example: Making a prediction on a single image from CIFAR-10
cifar10_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)
test_image, label = cifar10_testset[0]
test_image = test_image.unsqueeze(0)  # Add batch dimension

output = quantized_net(test_image, dataset='cifar10')
_, predicted = torch.max(output.data, 1)
print(f'Predicted label: {predicted.item()}, True label: {label}')

# Optional: Evaluate the model on a test dataset
correct = 0
total = 0
testloader_cifar10 = torch.utils.data.DataLoader(cifar10_testset, batch_size=64, shuffle=False)

with torch.no_grad():
    for data in testloader_cifar10:
        images, labels = data
        outputs = quantized_net(images, dataset='cifar10')
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the quantized network on CIFAR-10 test images: {100 * correct / total:.2f}%')
