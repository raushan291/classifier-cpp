import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import torch.nn.functional as F
import torchvision.models as models


class VGG(nn.Module):
    def __init__(self, num_classes, normalize_attn=False, dropout=None):
        super(VGG, self).__init__()
        net = models.vgg16_bn(pretrained=True)
        self.conv_block1 = nn.Sequential(*list(net.features.children())[0:6])
        self.conv_block2 = nn.Sequential(*list(net.features.children())[7:13])
        self.conv_block3 = nn.Sequential(*list(net.features.children())[14:23])
        self.conv_block4 = nn.Sequential(*list(net.features.children())[24:33])
        self.conv_block5 = nn.Sequential(*list(net.features.children())[34:43])
        self.pool = nn.AvgPool2d(7, stride=1)
        self.dpt = None
        if dropout is not None:
            self.dpt = nn.Dropout(dropout)
        self.cls = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    def forward(self, x):
        block1 = self.conv_block1(x)        # /1
        pool1 = F.max_pool2d(block1, 2, 2)  # /2
        block2 = self.conv_block2(pool1)    # /2
        pool2 = F.max_pool2d(block2, 2, 2)  # /4
        block3 = self.conv_block3(pool2)    # /4
        pool3 = F.max_pool2d(block3, 2, 2)  # /8
        block4 = self.conv_block4(pool3)    # /8
        pool4 = F.max_pool2d(block4, 2, 2)  # /16
        block5 = self.conv_block5(pool4)    # /16
        pool5 = F.max_pool2d(block5, 2, 2)  # /32
        N, __, __, __ = pool5.size()

        g = self.pool(pool5).view(N, 512)

        if self.dpt is not None:
            g = self.dpt(g)
        out = self.cls(g)
        
        return out


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def export_to_onnx(torch_model, batch_size=1):
    # Input to the model
    x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)

    torch_model = torch_model.to('cpu')

    # Export the model
    torch.onnx.export(torch_model,             # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    "vgg.onnx",                # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=12,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # for i, (images, labels) in enumerate(dataloader):
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # if i % 100 == 99:
        #     last_loss = running_loss / i  # loss per batch
        #     print('  batch {}/{} loss: {}'.format(i + 1, len(dataloader), last_loss))

    accuracy = 100 * correct / total
    return running_loss / len(dataloader), accuracy


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return running_loss / len(dataloader), accuracy


def test(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def main():
    # Data path
    DATA_PATH = "./mnist"

    # Model save path
    model_path = "./models/vgg.pth"

    best_model_path = os.path.join(os.path.dirname(model_path), os.path.splitext(os.path.basename(model_path))[0] + "_best.pth")
    last_model_path = os.path.join(os.path.dirname(model_path), os.path.splitext(os.path.basename(model_path))[0] + "_last.pth")

    # Device configuration
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    num_epochs = 3
    learning_rate = 1e-3
    batch_size = 64
    patience = 2

    image_size = (224, 224)  # should be a tuple
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Data augmentation and normalization
    transform_train = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Dataset
    train_dataset = datasets.ImageFolder(root=os.path.join(DATA_PATH, 'train'), transform=transform_train)
    val_dataset = datasets.ImageFolder(root=os.path.join(DATA_PATH, 'val'), transform=transform_val)

    # Data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # Model
    model = VGG(num_classes=10, dropout=0.5).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    best_accuracy = 0
    interval = 0
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

        if val_accuracy > best_accuracy:
            print(f"saving model at epoch {epoch+1}")
            torch.save(model.state_dict(), best_model_path)
            best_accuracy = val_accuracy
            interval = 0
        else:
            interval += 1

        if interval == patience:
            print(f"Early stopping. There is no improvement since last {patience} epoch.")
            break

    # Save last model
    print("saving last model")
    torch.save(model.state_dict(), last_model_path)

    print("Final evalution :")
    model.load_state_dict(torch.load(best_model_path))
    accuracy = test(model, val_loader, device)

    print(f"Accuracy on best model : {accuracy}")

    print(f"Exporting model to onnx..")
    export_to_onnx(model)
    print("pytorch to onnx conversion done.")


if __name__ == "__main__":
    main()
