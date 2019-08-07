import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision #data load preprocessing
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        #super A/B,C/D/E 순으로 class 상속이 이뤄질때 E가 나머지의 모든 특성을 갖도록 조정(상위 class를 모두 안넣어도 스스로 해결)
        #E까지의 모든 init을 실행
        self.fc1 = nn.Linear(in_features=28 * 28, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):   #activation function을 정의
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #input shape: [batch_size, 1, 28, 28]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=2)
        #output shape: [batch_size, 32, 14, 14]
        #(28 - 3 + 2*1) + 1 = 14 , 16개의 kernel
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2)
        #output shape: [batch_size, 64, 7, 7]
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2)
        #output shape: [batch_size, 128, 4, 4]
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2)
        #output shape: [batch_size, 256, 2, 2]
        self.linear = nn.Linear(256*2*2, 10)
        #chenel*h*w. 10개의 label
        #input이 kernel과 사이즈가 같기때문에 flatten을 시켜서 계산 (그냥 conv해도됨)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.shape[0], -1) # [Batch_size, 256*2*2]
        x = self.linear(x)
        return x

if __name__== '__main__' :
    # 현재 파일을 직접실행할때만 아래 코드를 실행 (test에선 실행하지 않음)

    MODEL = 'CNN' # 사용할 model선정

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])
    #input data를 torch tensor에 맞게, 값을 0과 1사이 값을 갖도록 그리고 정규분포를 따르도록 구성
    #data 전처리과정

    dataset = torchvision.datasets.MNIST(root='./datasets', train=True, transform=transform, download=True)

    #Train data만 구성하고 transform 형식에 맞도록 조정하여 download

    data_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=0, batch_size=32, shuffle=True)
    #batch size를 32로 shuffle시켜서 데이터셋 구성(CPU 할당을 0)

    if MODEL == 'CNN':
        from models import CNN
        models = CNN()
    elif MODEL == 'MLP':
        models = MLP()
    else:
        raise NotImplementedError(" You need to choose among [CNN , MLP]. ")



    #mlp = MLP()
    #인스던스

    loss = nn.CrossEntropyLoss()
    #activation function으로 CrossEntropy 함수 사용
    optim = torch.optim.Adam(models.parameters(), lr=2e-4, betas=(0.5, 0.99), eps=1e-8)
    #Optimazer로 Adam사용 상세내용은  leanring rate(lr) weight를 loss로 갱신할때 gradient값을 어느정도 보정
    # /batas 이전grad를 반영, 현재grad크기를 보정/ eps 계산할때쓰는것

    EPOCHS = 1
    #Epochs 를 n으로 설정
    total_step = 0
    list_loss = list()
    list_acc = list()

    for epoch in range(EPOCHS):
        for i, data in enumerate(data_loader):
            #Enumerate label도 같이 출력
            total_step = total_step + 1
            input, label = data[0], data[1]  #, data의 구성? 0=32,1,28*28 / 1= 32개의 label
            # input shape [batch size ,channel, height, width]
            input = input.view(input.shape[0], -1) if MODEL == 'MLP' else input
            # [batch size, channel*height*width][32, 28*28*1] 1차원 mlp에 넣기위해
            #view id를 보존/reshape: id 보존 x
            #if를 한줄로 쓸때 else는 반드시 기입

            classification_result = models(input)  # [32 10]

            l = loss(classification_result, label)
            list_loss.append(l.detach().item())  # item torch tensor 를 python의 형식으로 바꿔줌

            optim.zero_grad() #optim를 정하기 이전에 기존 gradient를 0로 설정(중복을 방지)
            l.backward()
            optim.step()
            print(l.detach().item())

    torch.save(models, '{}.pt'.format(MODEL)) #model 저장 키워드로 object를 받음

    plt.figure()
    plt.plot(range(len(list_loss)), list_loss, linestyle='--')
    plt.show()

