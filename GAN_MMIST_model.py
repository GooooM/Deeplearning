import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        model = [nn.Linear(in_features=100, out_features=128), nn.ReLU(inplace=True)]
        model += [nn.Linear(in_features=128, out_features=256), nn.ReLU(inplace=True)]
        model += [nn.Linear(in_features=256, out_features=28 * 28), nn.Sigmoid()]
        self.model = nn.Sequential(*model)

        # "The generator nets used a mixture of rectifier linear activations and sigmoid activations, while the
        #  discriminator net used maxout activations." - Generative Adversarial Networks

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        model = [Maxout(28 * 28, 256, dropout=False, k=5)]
        model += [Maxout(256, 128, dropout=True, k=5)]
        model += [nn.Linear(128, 1), nn.Sigmoid()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

 
#act ftn(dropout에 최적화)
#dropout: nod일부를 0값을 임의로 준다. =>
class Maxout(nn.Module):
    def __init__(self, in_features, out_features, k=2, dropout = Ture, p=0.5):
        super(Maxout, self).__init__()
        model = [nn.Dropout(p)]
        model += [nn.Linear(in_features, out_features * k)]

        self.model = nn.Sequential(*model)
        self.k = k

    def forward(self, x):
        x = self.model(x)
        x, _ = x.view(x.shape[0], x.shape[1] // self.k, self.k).max(-1)
        return x

