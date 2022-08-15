from typing import Union

from torch.types import (_int, _device)
from sdh_efficientnet.models.efficientnet_model import EfnModelLevel
from torch.utils.data import Dataset, DataLoader
from efficientnet_pytorch import EfficientNet
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import copy
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


class EfNet:
    modelLevel: EfnModelLevel = EfnModelLevel  # 사용자가 사용할 모델의 레벨
    usedModel: EfficientNet  # EfficientNet 에서 가져온 모델
    class_names: dict = {}  # { "0": class1, "1": class2 } // key 값은 0,1,2 ~~~

    __device: Union[_device, _int, str]
    __modelImageSize: int  # 모델의 이미지 사이즈
    __criterion: nn.CrossEntropyLoss
    __optimizer: optim.SGD
    __scheduler: optim.lr_scheduler.ReduceLROnPlateau
    __trainDataTransform: transforms.Compose
    __testDataTransform: transforms.Compose
    __batchSize: int = 16
    __randomSeed: int = 555

    def builder(self, level: EfnModelLevel, class_names):
        random.seed(self.__randomSeed)
        torch.manual_seed(self.__randomSeed)
        self.class_names = class_names
        self.modelLevel = level
        self.usedModel = EfficientNet.from_name(model_name=self.modelLevel["name"])

        self.__modelImageSize = EfficientNet.get_image_size(model_name=self.modelLevel["name"])
        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set gpu

        self.usedModel = self.usedModel.to(self.__device)  # 모델에 cuda or cpu 적용하기

        # [START] fine tuning
        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = optim.SGD(
            self.usedModel.parameters(),
            lr=0.01,  # 점차 낮춰가면서 학습이 잘되는지 테스트한다.
            momentum=0.9,
            weight_decay=1e-4
        )
        self.__scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.__optimizer, threshold=1, patience=1, mode='min')
        self.__transform_train_data()  # 학습데이터 변형 Transform 정보
        self.__transform_test_data()  # 테스트데이터 변형 Transform 정보

        if torch.cuda.is_available():
            self.usedModel = self.usedModel.cuda()
            self.__criterion = self.__criterion.cuda()
        # [END] fine tuning

    def load_trained_model(self, model_path):
        """ 모델 불러오기 """
        self.usedModel.load_state_dict(torch.load(model_path))

    def create_dataset(self, trainDataPath: str):
        """ 데이터셋 만들기 """
        dataset = torchvision.datasets.ImageFolder(
            root=trainDataPath,
            transform=self.__trainDataTransform
        )

        return self.__create_data_loader(self.__split_dataset(dataset))  # return DataLoader (train, val, test)

    def create_test_dataset(self, testDataPath: str):
        """ 데이터셋 만들기 """
        dataset = torchvision.datasets.ImageFolder(
            root=testDataPath,
            transform=self.__testDataTransform
        )
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.__batchSize, shuffle=False,
                                                 num_workers=4)
        return dataloader, len(dataloader)

    def imshow(self, inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    def train_model(self, data_loaders, saved_model_path, num_epochs=25):
        since = time.time()

        best_acc = 0.0
        train_loss, train_acc, valid_loss, valid_acc = [], [], [], []

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:
                if phase == 'train':
                    self.usedModel.train()  # Set model to training mode
                else:
                    self.usedModel.eval()  # Set model to training mode

                running_loss, running_corrects, num_cnt = 0.0, 0, 0

                # Iterate over data.
                for inputs, labels in data_loaders[phase]:
                    inputs = inputs.to(self.__device)
                    labels = labels.to(self.__device)

                    self.__optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.usedModel(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.__criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            self.__optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    num_cnt += len(labels)
                if phase == 'train':
                    self.__scheduler.step(loss)

                epoch_loss = float(running_loss / num_cnt)
                epoch_acc = float((running_corrects.double() / num_cnt).cpu() * 100)

                if phase == 'train':
                    train_loss.append(epoch_loss)
                    train_acc.append(epoch_acc)
                else:
                    valid_loss.append(epoch_loss)
                    valid_acc.append(epoch_acc)
                print('{} Loss: {:.2f} Acc: {:.1f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'valid' and epoch_acc > best_acc:
                    best_idx = epoch
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.usedModel.state_dict())
                    print('==> best model saved - %d / %.1f' % (best_idx, best_acc))
                    self.usedModel.load_state_dict(best_model_wts)
                    torch.save(self.usedModel.state_dict(), f=saved_model_path + "/" + str(epoch) + "_model.pt")
                    # torch.save(self.usedModel.state_dict(), f=saved_model_path + "/hi_model.pt")

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best valid Acc: %d - %.1f' % (best_idx, best_acc))
        print('model saved')
        self.__draw_train_graph(best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc)
        return best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc

    def test_and_visualize_model(self, test_dataloader, num_images=4):
        self.usedModel.eval()

        running_loss, running_corrects, num_cnt = 0.0, 0, 0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_dataloader):
                inputs = inputs.to(self.__device)
                labels = labels.to(self.__device)

                outputs = self.usedModel(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.__criterion(outputs, labels)  # batch의 평균 loss 출력

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                num_cnt += inputs.size(0)  # batch size

            test_loss = running_loss / num_cnt
            test_acc = running_corrects.double() / num_cnt
            print('test done : loss/acc : %.2f / %.1f' % (test_loss, test_acc * 100))

        # 예시 그림 plot
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_dataloader):
                inputs = inputs.to(self.__device)
                labels = labels.to(self.__device)

                outputs = self.usedModel(inputs)
                _, preds = torch.max(outputs, 1)
                for j in range(0, num_images):

                    if num_images % 2 == 1:
                        ax = plt.subplot((num_images // 2) + 1, 3, j + 1)
                    else:
                        ax = plt.subplot(num_images // 2, 3, j + 1)
                    ax.axis('off')
                    ax.set_title('(%s%%) %s : %s -> %s' % (
                        round(torch.softmax(outputs, dim=1)[j, 0].item() * 100, 2),
                        'True' if self.class_names[str(labels[j].cpu().numpy())] == self.class_names[
                            str(preds[j].cpu().numpy())] else 'False',
                        self.class_names[str(labels[j].cpu().numpy())],
                        self.class_names[str(preds[j].cpu().numpy())]))
                    self.imshow(inputs.cpu().data[j])

                if i == 0: break

    def __draw_train_graph(self, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc):
        """ 결과 그래프 그리기 """
        print('best model : %d - %1.f / %.1f' % (best_idx, valid_acc[best_idx], valid_loss[best_idx]))
        fig, ax1 = plt.subplots()

        ax1.plot(train_acc, 'b-')
        ax1.plot(valid_acc, 'r-')
        plt.plot(best_idx, valid_acc[best_idx], 'ro')
        ax1.set_xlabel('epoch')
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel('acc', color='k')
        ax1.tick_params('y', colors='k')

        ax2 = ax1.twinx()
        ax2.plot(train_loss, 'g-')
        ax2.plot(valid_loss, 'k-')
        plt.plot(best_idx, valid_loss[best_idx], 'ro')
        ax2.set_ylabel('loss', color='k')
        ax2.tick_params('y', colors='k')

        fig.tight_layout()
        plt.show()

    def __split_dataset(self, dataset):
        """ 데이터셋 나누기 """
        train_idx, tmp_idx = train_test_split(list(range(len(dataset))), test_size=0.15, random_state=self.__randomSeed)
        datasets = {}
        datasets['train'] = Subset(dataset, train_idx)
        tmp_dataset = Subset(dataset, tmp_idx)

        val_idx, test_idx = train_test_split(list(range(len(tmp_dataset))), test_size=0.3,
                                             random_state=self.__randomSeed)
        datasets['valid'] = Subset(tmp_dataset, val_idx)
        datasets['test'] = Subset(tmp_dataset, test_idx)
        return datasets

    def __create_data_loader(self, datasets):
        """ 데이터 로더 만들기 """
        dataloaders, batch_num = {}, {}
        dataloaders["train"] = torch.utils.data.DataLoader(datasets['train'],
                                                           batch_size=self.__batchSize, shuffle=True,
                                                           num_workers=4)
        dataloaders["valid"] = torch.utils.data.DataLoader(datasets['valid'],
                                                           batch_size=self.__batchSize, shuffle=False,
                                                           num_workers=4)
        dataloaders["test"] = torch.utils.data.DataLoader(datasets['test'],
                                                          batch_size=self.__batchSize, shuffle=False,
                                                          num_workers=4)

        batch_num['train'], batch_num['valid'], batch_num['test'] = len(dataloaders['train']), len(
            dataloaders['valid']), len(dataloaders['test'])
        print('batch_size : %d,  tvt : %d / %d / %d' % (
            self.__batchSize, batch_num['train'], batch_num['valid'], batch_num['test']))
        return dataloaders, batch_num

    def __transform_train_data(self):
        """ 학습 데이터 변형 Transform 정보 """
        self.__trainDataTransform = transforms.Compose([
            transforms.Resize(self.modelLevel["info"][2]),  # img size
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=2.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __transform_test_data(self):
        """ 테스트 데이터 변형 Transform 정보 """
        self.__testDataTransform = transforms.Compose([
            transforms.Resize(self.modelLevel["info"][2]),  # img size
            # transforms.CenterCrop((220,220)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
