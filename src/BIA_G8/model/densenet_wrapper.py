import torchvision
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.model = torchvision.models.densenet121(pretrained=True, progress=True)
        self.model.classifier = nn.Linear(in_features=self.model.classifier.in_features, out_features=1)

    def forward(self, x):
        return self.model(x)
