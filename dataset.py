import torchvision.datasets
import torchvision.transforms as transforms

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

trainData = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,  # if train=False, dataset -> val
    download=True,
    transform=transform
)
