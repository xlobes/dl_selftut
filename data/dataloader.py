import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def mnist_train_test_valid(batch_size, dataset_root):
  # 1) Transform: turn PIL images into tensors in [0,1] (scales pixel values from [0, 255] → [0.0, 1.0])
  transform = transforms.ToTensor()

  # 2) Download MNIST (store or look for the dataset in the folder ./mnist)
  # train=True → load the training split (60,000 images).
  # train=False → load the test split (10,000 images).
  # download=True → if the dataset doesn’t exist locally, download it from the internet.
  # transform=transform → applies the transformation defined earlier (ToTensor())
  trainval_ds = datasets.MNIST(root=dataset_root, train=True,  download=False, transform=transform)
  test_ds     = datasets.MNIST(root=dataset_root, train=False, download=False, transform=transform)

  # 3) Split the original 60k "trainval" into train (54k) and val (6k)
  # torch.manual_seed(0)                  # makes the split reproducible
  train_size = int(0.9 * len(trainval_ds))
  val_size   = len(trainval_ds) - train_size
  train_ds, val_ds = random_split(trainval_ds, [train_size, val_size])    # random_split(...) divides the dataset into two non-overlapping subsets

  # 4) Wrap in DataLoaders: Wraps a dataset and provides mini-batches of tensors for training or testing
  # shuffle=True → randomizes the order of samples each epoch (useful only for training).
  # shuffle=False → keeps samples in fixed order (common for validation and test).
  train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
  val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
  test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

  # The function returns the three prepared data loaders so they can be used for model training and evaluation.
  return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example Usage:
    train_loader, val_loader, test_loader = mnist_train_test_valid(batch_size=10, dataset_root='../../datasets/mnist/')    #calls the function
    x, y = next(iter(train_loader))          # next(...)gets the first batch     iter(train_loader): creates an iterator over batches.   x: batch of images, shape [batch_size, 1, 28, 28].  y: batch of integer labels, shape [batch_size].
    print("Train batch:", x.shape, y.shape)  # e.g., torch.Size([64, 1, 28, 28]) torch.Size([64])
    x, y = next(iter(val_loader))
    print("Val batch:  ", x.shape, y.shape)
    x, y = next(iter(test_loader))
    print("Test batch: ", x.shape, y.shape)