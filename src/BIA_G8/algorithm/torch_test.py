import skimage.transform
import torch.optim
import torch.utils.data as tud
from torch import nn

from BIA_G8 import get_lh
from BIA_G8.covid_helper import covid_dataset

_lh = get_lh(__name__)


class MyModule(nn.Module):
    """
    Copy-and-paste from <https://blog.csdn.net/qq_45588019/article/details/120935828>
    """

    def __init__(
            self,
            n_features: int,
            n_classes: int
    ):
        super(MyModule, self).__init__()
        ksp = {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1
        }
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=16,
                **ksp
            ),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=16,
                out_channels=32,
                **ksp
            ),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                **ksp
            ),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0
            ),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(
            in_features=n_features // 4,
            out_features=64
        )
        self.mlp2 = torch.nn.Linear(
            in_features=64,
            out_features=n_classes
        )

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(batch_size, -1))
        x = self.mlp2(x)
        return x


if __name__ == '__main__':
    data_len = 300
    dataset = covid_dataset.CovidDataSet.parallel_from_directory(
        "/media/yuzj/BUP/covid19-database-np",
        size=data_len
    ).parallel_apply(
        lambda img: skimage.transform.resize(
            img,
            (64, 64)
        )
    )
    train_data_loader = tud.DataLoader(dataset[0:80].torch_dataset, batch_size=16, shuffle=True)
    test_data_loader = tud.DataLoader(dataset[80:120].torch_dataset, batch_size=16, shuffle=True)
    loss_count = []
    model = MyModule(
        n_features=64 * 64,
        n_classes=3
    )
    loss_func = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(20):
        for i, (x_train, y_train) in enumerate(train_data_loader):
            y_train_pred_prob = model(x_train)
            y_train_pred = torch.argmax(y_train_pred_prob, dim=-1)
            loss = loss_func(y_train_pred_prob, y_train)
            opt.zero_grad()
            loss.backward()
            opt.step()
            accu = (torch.sum(y_train_pred == y_train) * 100 / y_train.shape[0]).item()
            if i % 10 == 0:
                print(f"Epoch {epoch} batch {i}: accuracy {accu:.2f}")
    for i, (x_test, y_test) in enumerate(test_data_loader):
        y_test_pred_prob = model(x_test)
        y_test_pred = torch.argmax(y_test_pred_prob, dim=-1)
        accu = (torch.sum(y_test_pred == y_test) * 100 / y_test.shape[0]).item()
        print(f"Predict batch {i}: accuracy {accu:.2f}")
