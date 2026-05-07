from torchvision import datasets

datasets.CIFAR10(root="./data", train=True, download=True)
datasets.CIFAR10(root="./data", train=False, download=True)

print("Done downloading CIFAR-10")