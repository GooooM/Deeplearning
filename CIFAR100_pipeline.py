#사이파10 의 전처리 파이프라인 좌우플립, normalization

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor, Normalize, Compose, RandomHorizontalFlip, Pad, RandomCrop #Flip, pad crop 모두 PIL을 받도록되어있다 => Totensor는 가장 마지막에
import numpy as np

class CustomCIFAR100(Dataset):
    def __init__(self):
        super(CustomCIFAR100, self).__init__()
        self.cifar_100 = CIFAR100(root='./datasets', train=True, download=True)

        tensors = list()
        for i in range(len(self.cifar_100)):
            tensors.append(ToTensor()(self.cifar_100[i][0]).numpy()) # ToTensor => hwc에서 chw 0~1사이값을 만들어줌 #Totensor에서 계산이 안되므로 np사용
            #break #중간에 데이터를 확인하기위하여
        mean = np.mean(tensors, axis=(0, 2, 3))
        std = np.std(tensors, axis=(0, 2, 3))
        #axis 0= 데이터갯수 1 = h 2 =w 3= channel Totensor가 아닐때 (PIL형식)

        transform = [RandomHorizontalFlip()]
        transform += [Pad(4), RandomCrop(32)]
        transform += [ToTensor(), Normalize(mean=mean, std=std)]
        self.transform = Compose(transform)



        #print('mean: {}, std: {}'.format(mean, std))
    def __getitem__(self, index):
        #인덱싱 리스트안 요소를 접근하도록
        tensor, label = self.transform(self.cifar_100[index][0]), self.cifar_100[index][1] #index 0=이미지, 1=라벨
        return tensor, label

    def __len__(self):
        return len(self.cifar_100)

if __name__ == '__main__':
    dataset = CustomCIFAR100()
    print(dataset[0]) # 0의 의미?