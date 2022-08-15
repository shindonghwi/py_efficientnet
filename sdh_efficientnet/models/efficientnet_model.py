from enum import Enum

"""
# name: 모델 이름
# info: (width,depth,res,dropout)

"""


class EfnModelLevel(dict, Enum):
    version_0 = {'name': 'efficientnet-b0', 'info': (1.0, 1.0, 224, 0.2)}
    version_1 = {'name': 'efficientnet-b1', 'info': (1.0, 1.1, 240, 0.2)}
    version_2 = {'name': 'efficientnet-b2', 'info': (1.1, 1.2, 260, 0.3)}
    version_3 = {'name': 'efficientnet-b3', 'info': (1.2, 1.4, 300, 0.3)}
    version_4 = {'name': 'efficientnet-b4', 'info': (1.4, 1.8, 380, 0.4)}
    version_5 = {'name': 'efficientnet-b5', 'info': (1.6, 2.2, 456, 0.4)}
    version_6 = {'name': 'efficientnet-b6', 'info': (1.8, 2.6, 528, 0.5)}
    version_7 = {'name': 'efficientnet-b7', 'info': (2.0, 3.1, 600, 0.5)}
    version_8 = {'name': 'efficientnet-b8', 'info': (2.2, 3.6, 672, 0.5)}
    version_l2 = {'name': 'efficientnet-l2', 'info': (4.3, 5.3, 800, 0.5)}  # 사전 훈련된 가중치 적용이 안되어 있는 네트워크
