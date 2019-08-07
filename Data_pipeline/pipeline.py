#data loader class 생성
import os
from torch.utils.data import Dataset
import random
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, Grayscale
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root, crop_size=0, flip=False): #경로 #좌우 바꾸기 #패치(image crop) #흑백 인자를 갖도록 설정 #다른 함수에서도 접근권한이 생김 #defult가 0, False로 둠
        super(CustomDataset, self).__init__()
        self.root = root
        self.list_paths = os.listdir(root)  #root 내의 파일을 list로 불러온다.
        self.crop_size = crop_size
        self.flip = flip

    def __getitem__(self, index): #magic method 인덱싱을 해준다. 정의가 안되어있으면 인덱싱이 불가 #그냥 괄호로 할수있도록 해주는 magic method 그냥 함수로 만들어된다. a.indexing()으로 써야함
        image = Image.open(os.path.join(self.root, self.list_paths[index])) #위에서 정의된 list_path로 불러오는 이미지들을 index
        #join으로 경로+이름으로 합쳐서 image.open을 쓸 수있도록
        #image.show()

        list_transforms = list()
        if self.crop_size > 0:
            list_transforms.append(RandomCrop((self.crop_size, self.crop_size)))
        if self.flip:
            coin = random.random() > 0.5 #크면 True 작으면 False를 돌려줌
            if coin: #if coin== True와 같은 의미
                list_transforms.append(RandomHorizontalFlip())

        transfroms = Compose(list_transforms)

        image = transfroms(image)

        input_image = Grayscale(num_output_channels=1)(image) # 흑백이미지 생성 0~255값을 갖음

        input_tensor = ToTensor()(input_image) #ToTensor도 class라 ()를 써야 인스턴스가 생성 Totorch로 바꾸고 0~1사이의 값을 갖음
        target_tensor = ToTensor()(image)

        input_tensor = Normalize(mean=[0.5], std=[0.5])(input_tensor)
        target_tensor = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(target_tensor)

        return input_tensor, target_tensor

    def __len__(self): #custum class에 반드시 들어아야할 method (init, getitem, len)
        return len(self.list_paths) #Element의 갯수를 출력하도록

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import numpy as np
    dir_image = './datasets/IU'
    list_paths = os.listdir(dir_image)
    dataset = CustomDataset(root=dir_image, crop_size=128, flip=True)
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0)

    for input, target in data_loader:
        input_image_np = np.array(input[0, 0]) #[B, C, H, W]중에 B와 C를 없애서 이미지로 만들려고 input[0, 0]을 한다.
        input_image_np -= input_image_np.min()
        input_image_np /= input_image_np.max()
        input_image_np *= 255.0 #0~1사이값을 갖는 것을 0~255값을갖도록만든다
        input_image_np = input_image_np.astype(np.uint8)
        input_image = Image.fromarray((input_image_np))
        input_image.show()

        target_image_np = np.array(target[0])
        target_image_np -= target_image_np.min()
        target_image_np /= target_image_np.max()
        target_image_np *= 255.0
        target_image_np = target_image_np.astype(np.uint8)
        target_image = Image.fromarray(target_image_np.transpose(1, 2, 0), mode='RGB') #H,W B순으로 바꿔줌 흑백은 채널을 컴퓨터가 못받아드림
        target_image.show()

        break

