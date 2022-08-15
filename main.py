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
        trainDataPath="./frame/data"
    )

    # 학습 시작
    instance.train_model(
        data_loaders=data_loaders,
        saved_model_path='./saved_model',
        num_epochs=100
    )

    # # train check
    # num_show_img = 5
    # inputs, classes = next(iter(data_loaders['train']))
    # out = torchvision.utils.make_grid(inputs[:num_show_img])  # batch의 이미지를 오려부친다
    # instance.imshow(out, title=[class_names[str(int(x))] for x in classes[:num_show_img]])
    # # valid check
    # inputs, classes = next(iter(data_loaders['valid']))
    # out = torchvision.utils.make_grid(inputs[:num_show_img])  # batch의 이미지를 오려부친다
    # instance.imshow(out, title=[class_names[str(int(x))] for x in classes[:num_show_img]])
    # # test check
    # inputs, classes = next(iter(data_loaders['test']))
    # out = torchvision.utils.make_grid(inputs[:num_show_img])  # batch의 이미지를 오려부친다
    # instance.imshow(out, title=[class_names[str(int(x))] for x in classes[:num_show_img]])

    """ 테스트 """
    # test_dataloader, test_batch_num = instance.create_test_dataset(
    #     testDataPath='./frame/test'
    # )
    # instance.load_trained_model(model_path='./saved_model/23_model.pt')
    # instance.test_and_visualize_model(test_dataloader, num_images=7)
