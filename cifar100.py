import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
from torch.optim.lr_scheduler import StepLR
from PIL import Image

transformation = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 隨機水平翻轉
    transforms.RandomRotation(10),  # 隨機旋轉，範圍為 -10 到 10 度
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = torchvision.datasets.CIFAR100("data", train=True, transform=transformation, download=True)

test_data = torchvision.datasets.CIFAR100("data", train=False, transform=transformation, download=True)
train_data_size = len(train_data)
test_data_size = len(test_data)
print("test data set size:",test_data_size)
print("train data set size:",train_data_size)

batch_size = 128
num_of_label = 100

train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ("apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle", "bowl", "boy",
           "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
           "cloud", "cockroach", "couch", "cra", "crocodile", "cup", "dinosaur", "dolphin", "elephant", "flatfish", "forest",
           "fox", "girl", "hamster", "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster",
           "man", "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
           "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose",
           "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper",
           "table", "tank", "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm")

class CIFAR100(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout2d(0.25),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256*2*2, 256),
            nn.Linear(256, 100)
        )

    def forward(self, x):
        x = self.model(x)
        return x
model = CIFAR100()
'''
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=stride , padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1 , padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        # 如果輸入和輸出的通道數不一致，需要添加一個額外的 1x1 卷積層來進行映射
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 定義 ResNet 模型
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 建立 ResNet-18 模型
def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

model = resnet18()
'''
loss_fn = nn.CrossEntropyLoss()

learning_rate = 1e-3 #0.001
weight_decay = 1e-4
optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

total_train_step = 0
total_test_step = 0
epoch = 50
model = model.to(device)
patience = 8
best_valid_loss = float('inf')
current_patience = 0
best_model = 0
train_loss = []
test_loss = []
start_time = time.time()
scheduler = StepLR(optim, step_size=8, gamma=0.1)
for i in range(epoch):
    print(f"train round{i+1}/{epoch}")
    model.train()
    running_train_loss = 0.0
    train_loss.append(0)
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)

        optim.zero_grad()
        output = model(imgs)
        loss = loss_fn(output, targets)

        loss.backward()
        optim.step()
        train_loss[-1] += loss.item()

        running_train_loss += loss.item() * imgs.size(0)
        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            total_time = end_time-start_time
            print(f"time:{total_time:.2f}")
            print(f"train ={total_train_step}, loss={loss.item():.5f}")
    scheduler.step()
    # 測試步驟開始
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    running_valid_loss = 0
    with torch.no_grad():
        test_loss.append(0)
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = model(imgs)
            loss = loss_fn(output, targets)
            test_loss[-1] += loss.item()
            running_valid_loss += loss.item() * imgs.size(0)

            total_test_loss = total_test_loss + loss.item()
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy =total_accuracy + accuracy
    epoch_valid_loss = running_valid_loss / test_data_size
    print(f"整體測試集的loss:{total_test_loss:.2f}")
    print(f"整體測試集的正確率{total_accuracy/test_data_size:.3f}")

    total_test_loss += 1
    if epoch_valid_loss < best_valid_loss:
        best_valid_loss = epoch_valid_loss
        current_patience = 0
        best_model = i+1
        torch.save(model, f"cifar100_{i + 1}.pth")
        print("model has been saved")
    else:
        current_patience += 1
        if current_patience >= patience:
            print(f"Early stopping after {i + 1} epochs.")
            break

    # if i%10 == 9:
    #   torch.save(model, f"cifar10_{i+1}.pth")
    #   print("model has been saved")

plt.plot(train_loss, label="Training loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(test_loss, label="Testing loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model = torch.load(f"cifar100_{best_model}.pth", map_location=torch.device("cpu"))

def imageshow(img):
  img = img/2 + 0.5
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show

def testBatch():
  images, labels = next(iter(test_dataloader))
  imageshow(torchvision.utils.make_grid(images))
  print("Real Label:", ''.join("%5s"%classes[labels[j]]
              for j in range(batch_size)))
  outputs = model(images)
  _,predicted = torch.max(outputs, 1)
  print("Predicted", ''.join("%5s"%classes[labels[j]]
              for j in range(batch_size)))

testBatch()
'''
image_path ="./test image/ai_cat.jpeg"
image =Image.open(image_path)
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),torchvision.transforms.ToTensor()])

image = transform(image)
# model = torch.load(f"cifar100_{best_model}.pth", map_location=torch.device("cpu"))
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()#transfer to test mode
with torch.no_grad():
  output = model(image)
print(output)
print(classes[output.argmax(1)])
print(output.argmax(1))
'''