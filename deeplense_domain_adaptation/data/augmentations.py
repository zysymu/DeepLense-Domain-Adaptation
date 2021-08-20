from torchvision import transforms

def supervised_augmentations():
    train_transform = transforms.Compose([transforms.Resize(150),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          ])

    test_transform = transforms.Compose([transforms.Resize(150),
                                        ])

    return train_transform, test_transform