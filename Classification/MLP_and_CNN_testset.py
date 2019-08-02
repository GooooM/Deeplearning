import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision
from models import MLP, CNN

if __name__ == '__main__':


    #Input pipeline
    transform = Compose([ToTensor(), Normalize([0.5], [0.5])])
    dataset = MNIST('./datasets', False, transform, download=True)
    data_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=False, num_workers=0)

    # Define model
    MODEL = 'CNN'
    if MODEL == 'CNN':
        models = CNN()
    elif MODEL == 'MLP':
        models = MLP()
    else:
        raise NotImplementedError(" You need to choose among [CNN , MLP]. ")

    # load model
    trained_model = torch.load('{}.pt'.format(MODEL)) # 저장된 model 를 load
    state_dict = trained_model.state_dict()  # model에서 weight를 추출하기위한 함수
    # state.dict() dictionary: key/ value : key로 value를 부른다 (인덱싱 위치를 알 필요가 없다)
    # print(state_dict.key()) 하면 key를 알 수 있음
    # 반대로 모델을 저장할때 state.dict()를 선언하여 다음에 모델을 부를 때 keyword와 weight만 불러오는게 좋다.
    # wight : value, ??? : key key-> model 정의에서 볼수있다 (con or fc)
    # trained_model.state_dict() 그냥 실행하면 MLP 부분에서 train이 진행되서 loss가 간접적으로 실행 train에서 if문으로 수정

    models.load_state_dict(state_dict)
    #저장된 weight를 불러온다 model 내의 같은 layer로 저장해야 돌아간다 fc1,2,3.

    nb_correct_answers = 0
    for data in data_loader:
        input, label = data[0], data[1]
        # input = input.view(input.shape[0], -1)
        classification_results = models(input)
        nb_correct_answers += torch.eq(classification_results.argmax(dim=1), label).sum()
        # dim=1 16X10에서 각 줄마다 10개중 하나를 뽑도록 해서 16개의 값을 나오도록
        # dim=0 16X10에서 각 열마다 16개중 하나를 뽑도록 해서 10개의 값을 나오도록
        # torcheq : 각리스트들의 엘리먼트를 비교해서 같으면 1을 return
        #argmax 가장 큰 확률 값의 인덱스를 표출
        #torch.eq 두값 armax, laber가 맞으면 1 아니면 0
        #sum batchzise 16이라 16개의 값을 표출하는것을 다 더함 min0, max16

    print('Average acc.:{}%.'.format(float(nb_correct_answers) / len(dataset) * 100))
