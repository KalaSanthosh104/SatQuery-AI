import torchvision.transforms as transforms

cdvqa_transform = transforms.Compose([
    transforms.Resize(
        (256, 256)
    ),
    transforms.ToTensor()
])