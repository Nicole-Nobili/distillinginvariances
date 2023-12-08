# train.py
import argparse
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models import SimpleCNN, LeNet

def parse_args():
    parser = argparse.ArgumentParser(description='Train a simple CNN on MNIST')
    parser.add_argument('--model', type=str, default='SimpleCNN', choices=['SimpleCNN, LeNet'],
                        help='Choose the CNN model architecture'),
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_conv_layers', type=int, default=2)
    parser.add_argument('--temperature', type=float, default=1, help='Define temperature'),
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--model_name', type=str, default='simple_cnn_model.pth', help='Name of the saved model file')
    return parser.parse_args()

def train(model, num_epochs, model_name):
    # ... (rest of your code remains unchanged)

    # Save the trained model
    torch.save(model.state_dict(), model_name)
    print(f"Model saved as {model_name}!")

if __name__ == "__main__":
    args = parse_args()

    # Initialize the model based on the chosen architecture
    if args.model == 'SimpleCNN':
        model = SimpleCNN(in_channels=1, num_classes=10, num_conv_layers=3)
    else:
        raise ValueError(f"Unsupported model architecture: {args.model}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Call the training function with the provided arguments
    train(model, args.num_epochs, args.model_name)

    # Testing the model (remaining code unchanged)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Test Accuracy: {accuracy:.4f}')