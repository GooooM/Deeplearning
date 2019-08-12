import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        act = nn.ReLU(inplace=True)
        model = [nn.Linear(100, 128), act]
        #inplace 메모리주소에 할당된 variable을 덮어쓴다.
        # += inplace operation
        model += [nn.Linear(128, 256), act]
        model += [nn.Linear(256, 512), act]
        model += [nn.Linear(512, 1*28*28), nn.Tanh()] #Tanh : -1,1사이의 값을 갖도록
        self.model = nn.Sequential(*model)

        # Sequential은 모델을 정의한 순서대로 input을 통과시켜줌, Module을 받도록되어있음
        # *model: (리스트 (container) 안의 값만 뽑도록 해줌) Class안에 모듈로 이뤄진 리스트 model에서 모듈만 받도록 (*의 역할)
        # x = F.tanh(self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(F.relu(self.fc1(x))))))) 를 대신해줌.
        # 결론적으로 self.fc1 = nn.Linear(100,128) 와 x = F.relu(self.fc1(x))들을 안써도 된다.
    def forward(self, x):
        # x = F.tanh(self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(F.relu(self.fc1(x)))))))에서 x를 넣어주는 역할
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        act = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        model = [nn.Linear(1*28*28, 512), act]
        model += [nn.Linear(512, 256), act]
        model += [nn.Linear(256, 128), act]
        model += [nn.Linear(128, 1), nn.Sigmoid()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    from torchvision.transforms import Compose, ToTensor, Normalize
    from torchvision.datasets import MNIST
    from torch.utils.data import DataLoader
    import torch
    import  matplotlib.pyplot as plt
    from torchvision.utils import save_image

    LR = 2e-4
    BETA1, BETA2 = 0.5, 0.99
    EPOCHS = 3
    BATCH_SIZE = 16
    LATENT_DIM = 100
    ITER_DISPLAY = 10

    transform = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])
    dataset = MNIST(root='./datasets', train=True, transform=transform, download=True)
    data_loader = DataLoader(dataset=dataset, num_workers=0, batch_size=BATCH_SIZE, shuffle=True)

    D = Discriminator()
    G = Generator()

    criterion = nn.BCELoss()
    #Binary cross entropy loss

    optim_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(BETA1, BETA2))
    optim_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(BETA1, BETA2))

    total_step = 0
    for epoch in range(EPOCHS):
        for data in data_loader:
            real, label = data[0], data[1]
            real = real.view(BATCH_SIZE, -1)

            z = torch.randn(BATCH_SIZE, LATENT_DIM)
            #random한 100개의 값을 배치사이즈에 맞는 표준정규분포에 맞게 생성
            fake = G(z) # BATCH_SIZE * 784

            validity_fake = D(fake.detach())
            #detach D를 업데이트할때 G의 weight들에게도 gradient를 나눠줌 실제로 영향은 없지만 시간소모를 안하도록
            validity_real = D(real)

            loss_D = (criterion(validity_fake, torch.zeros_like(validity_fake))
                      + criterion(validity_real, torch.ones_like(validity_fake))) * 0.5
            #ones_like, zeros_like : 뒤에 받는 형태로 0,1값을 산출

            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

            validity_fake = D(fake)

            loss_G = criterion(validity_fake, torch.ones_like(validity_fake))

            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

            if total_step % ITER_DISPLAY == 0:
                # % 두 사이의 나머지값
                fake_image = fake.detach().view(BATCH_SIZE, 1, 28, 28)
                real_image = real.view(BATCH_SIZE, 1, 28, 28)
                save_image(fake_image, '{}_fake.png'.format(epoch + 1), nrow=4, normalize=True)
                save_image(real_image, '{}_real.png'.format(epoch + 1), nrow=4, normalize=True)
                #nrow: 배치사이즈를 해당값으로 나눠서 행렬로 row당 몇개의 이미지가 있는지를 알려줌 나머지는 버림
