import os
import torch
import math
import time
import skimage as ski
from pathlib import Path
import numpy as np
from torch import nn
from torch.optim import SGD
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split

DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
SAVE_DIR = Path(__file__).parent / 'out'
np.random.seed(int(time.time() * 1e6) % 2**31)

config = {
    'max_epochs': 8,
    'batch_size': 50,
    'weight_decay': 1e-2,
    'lr_policy': {1: {'lr': 1e-1}, 3: {'lr': 1e-2}, 5: {'lr': 1e-3}, 7: {'lr': 1e-4}},
    'verbose': True
}

def draw_conv_filters(epoch, conv_layer, save_dir):
    w = conv_layer.weight.clone().data.cpu().numpy()
    
    num_filters = w.shape[0]
    C = w.shape[1]
    k = w.shape[2]
    
    w -= w.min()
    w /= w.max()
    
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols - 1) * border
    height = rows * k + (rows - 1) * border
    
    for i in range(C):
        img = np.zeros([height, width])
        for j in range(num_filters):
            r = int(j / cols) * (k + border)
            c = int(j % cols) * (k + border)
            img[r:r+k, c:c+k] = w[j, i]
        
        filename = f'filter_epoch_{epoch:02d}_channel_{i:03d}.png'
        ski.io.imsave(os.path.join(save_dir, filename), ski.img_as_ubyte(img))

class CovolutionalModel(nn.Module):
  def __init__(self, in_channels, image_height, image_width, conv1_channels, conv2_channels, fc1_width, class_count):
    super().__init__()
    # conv and pooling layers
    self.conv1 = nn.Conv2d(in_channels, conv1_channels, kernel_size=5, stride=1, padding=2, bias=True)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=5, stride=1, padding=2, bias=True)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    fc1_in = conv2_channels * math.floor(math.floor(image_width / 2) / 2) * math.floor(math.floor(image_height / 2) / 2)
    # fully connected layers
    self.fc1 = nn.Linear(fc1_in, fc1_width, bias=True)
    self.fc_logits = nn.Linear(fc1_width, class_count, bias=True)

    # parameters already initialized with Conv2d and Linear
    # but we can redifine them
    self.reset_parameters()

  def reset_parameters(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear) and m is not self.fc_logits:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
    self.fc_logits.reset_parameters()

  def forward(self, x):
    h = self.conv1(x)
    h = self.pool1(h)
    h = torch.relu(h)
    h = self.conv2(h)
    h = self.pool2(h)
    h = h.view(h.shape[0], -1)
    h = self.fc1(h)
    h = torch.relu(h)
    logits = self.fc_logits(h)
    return logits


def train(mnist_train_set, mnist_valid_set, model, config):
    train_loader = DataLoader(mnist_train_set, batch_size=config['batch_size'], shuffle=True)
    valid_loader = DataLoader(mnist_valid_set, batch_size=config['batch_size'], shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=config['lr_policy'][1]['lr'], weight_decay=config['weight_decay'])
    
    for epoch in range(1, config['max_epochs'] + 1):
      if epoch in config['lr_policy']:
        for param_group in optimizer.param_groups:
          param_group['lr'] = config['lr_policy'][epoch]['lr']
                
      model.train()
      total_loss = 0
      correct = 0
      total = 0
      for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
            
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
      draw_conv_filters(epoch, model.conv1, SAVE_DIR)
      print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}, Accuracy: {correct / total * 100}%")
        
      # validation
      model.eval()
      valid_correct = 0
      valid_total = 0
      with torch.no_grad():
        for inputs, labels in valid_loader:
          outputs = model(inputs)
          _, predicted = torch.max(outputs.data, 1)
          valid_total += labels.size(0)
          valid_correct += (predicted == labels).sum().item()
                
      print(f"Validation Accuracy: {valid_correct / valid_total * 100}%")


def evaluate(model: CovolutionalModel, dataset: DataLoader, loss_function):
    current_loss = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for it, (x, y) in enumerate(dataset):
            logits = model.forward(x)
            loss = loss_function(logits, y)

            logits = nn.functional.softmax(logits, dim=1)
            for i, p in enumerate(logits):
                if y[i] == torch.max(p.data, 0)[1]:
                    correct += 1

            current_loss += float(loss)

    loss = current_loss / len(dataset)
    accuracy = correct / len(dataset.dataset)

    print(f"Test loss >  {loss:.06}, accuracy: {accuracy * 100:.03}%")


if __name__ == '__main__':
    mnist_train_set = MNIST(DATA_DIR, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    mnist_test_set = MNIST(DATA_DIR, train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    mnist_train_set, mnist_valid_set = random_split(mnist_train_set, [int(0.8 * len(mnist_train_set)), int(0.2 * len(mnist_train_set))])

    test_dl = DataLoader(dataset=mnist_test_set, batch_size=config['batch_size'], shuffle=True)

    model = CovolutionalModel(1, 28, 28, 16, 32, 512, 10)
    loss_function = nn.CrossEntropyLoss()

    train(mnist_train_set, mnist_valid_set, model, config)
    evaluate(model, test_dl, loss_function)
