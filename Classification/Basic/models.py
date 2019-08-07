import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        # init은 그 아래 코드들을 해당 class를 인스던스를 만들 때 마다 실행되도록 하는것
        super(MLP, self).__init__()
        # super A/B,C/D/E 순으로 class 상속이 이뤄질때 E가 나머지의 모든 특성을 갖도록 조정(상위 class를 모두 안넣어도 스스로 해결)
        # E까지의 모든 init을 실행
        self.fc1 = nn.Linear(in_features=28 * 28, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):  # activation function을 정의
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # input shape: [batch_size, 1, 28, 28]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=2)
        # output shape: [batch_size, 32, 14, 14]
        # (28 - 3 + 2*1) + 1 = 14 , 16개의 kernel
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2)
        # output shape: [batch_size, 64, 7, 7]
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2)
        # output shape: [batch_size, 128, 4, 4]
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2)
        # output shape: [batch_size, 256, 2, 2]
        self.linear = nn.Linear(256 * 2 * 2, 10)
        # chenel*h*w. 10개의 label
        # input이 kernel과 사이즈가 같기때문에 flatten을 시켜서 계산 (그냥 conv해도됨)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.shape[0], -1)  # [Batch_size, 256*2*2]
        x = self.linear(x)
        return x