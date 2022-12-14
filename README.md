### 1. Install PyTorch EfficientNet
```
git clone https://github.com/lukemelas/EfficientNet-PyTorch
cd EfficientNet-Pytorch
pip install -e .
```

or

```
pip install efficientnet_pytorch
```

---------------------

### 2. Git Clone [ shindonghwi / py_efficientnet ]
```
git clone https://github.com/shindonghwi/py_efficientnet/tree/main
```

### 3. Install pip requirement.txt
```
pip install -r requirements.txt
```

---------------------
### 4. Create Folder ( train, test, model )
```
root
  - main.py
  - data ( is new )
      - train
          - class 0 ( ex. dog ) 
          - class 1 ( ex. cat )
          - ... 
      - test
          - img
              - 1.jpg
              - 2.jpg
              - 3.jpg
              - ...
  - saved_model ( is new )
```

### 5. Train And Test

Common
```
from sdh_efficientnet.EfficientNet import EfNet

class_names = {
    "0": "CAT",
    "1": "DOG",
}

if __name__ == '__main__':
    instance = EfNet()

    instance.builder(
        level=instance.modelLevel.version_3,  # model level 설정 [ 0 ~ 8 or l2 ]
        class_names=class_names
    )

    data_loaders, batch_nums = instance.create_dataset(
        trainDataPath="./data/train"
    )
```

Train ( add code below)
```
instance.train_model(
    data_loaders=data_loaders,
    saved_model_path='./saved_model',
    num_epochs=100
)
```

Test ( add code below)
```
test_dataloader, test_batch_num = instance.create_test_dataset(
    testDataPath='./data/test'
)
instance.load_trained_model(model_path=last saved model path)
instance.test_and_visualize_model(test_dataloader, num_images=7)
```