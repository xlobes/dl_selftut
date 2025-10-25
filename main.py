import torch
import torch.nn as nn
from data.dataloader import mnist_train_test_valid
from nn.logreg import LogisticRegression
from nn.lenet import LeNet5

def train(f_model, loss_funcion, optimizer_function, device):
    # --- 3) Train for 10 epochs on the training set ---
    for epoch in range(1, 11):
        f_model.train()               # Puts the model in training mode (logistic regression)
        sum_loss_of_epoch = 0         # total loss accumulated over all batches.
        sum_accuracy_of_epoch = 0     # total number of correctly classified examples.
        total_check = 0               # (unused in this code) often tracks number of samples processed.

        for batch_i in train_loader:  # Iterates over each mini-batch from the training loader
            (x, y) = batch_i
            x, y = x.to(device), y.to(device)

            optimizer_function.zero_grad()      # Clears previous gradients stored in the optimizer.
                                                # "PyTorch accumulates gradients by default, so you must reset them before each new backward pass."
            y_hat = f_model.forward(x)          # f_model computes logits (raw scores) for each class.Output shape: [64, 10] (10 class scores per image).
            loss = loss_funcion(y_hat, y)       # Computes the loss between predictions y_hat and true labels y (average loss for this batch)
            loss.backward()                     # backpropagation
            optimizer_function.step()           # Updates model weights using the gradients computed above

            # keep loss for print
            sum_loss_of_epoch = sum_loss_of_epoch + loss.item()

            # accuracy compute for printhhh
            y_pred = y_hat.argmax(dim=1)
            number_of_accuracy = (y_pred == y).sum().item()
            sum_accuracy_of_epoch = sum_accuracy_of_epoch + number_of_accuracy

        # end of one epoch 

        # {epoch:02d} → formats epoch number with 2 digits (e.g., 01, 02, …).
        # :.4f → format this floating-point number with exactly 4 digits after the decimal point.
        print(f"Epoch {epoch:02d} | train loss: {sum_loss_of_epoch/number_of_examples_in_trainset:.4f} | train acc: {sum_accuracy_of_epoch/number_of_examples_in_trainset:.4f}")

if __name__ == "__main__":
    batch_size = 64
    train_loader, _, test_loader = mnist_train_test_valid(batch_size, dataset_root='../datasets/mnist/')
    number_of_examples_in_trainset = len(train_loader) * batch_size   # number of examples in the trainin set 843 * 64 = 53,952 ≈ 54,000 images
    print(f'There are {len(train_loader)} number of batches in train set, and batch_size is {batch_size}.')
    print(f'The total number of examples in train set is {number_of_examples_in_trainset}')
    print("------------------------------")

    device = "cuda"
    model = 'LogReg' # 'LeNet5'

    if model == 'LogReg':
        f_model = LogisticRegression(num_classes=10).to(device)  # 10 classes for MNIST
    elif model == 'LeNet5':
        f_model = LeNet5(num_classes=10).to(device)

    # loss function is a criteria to measure the difference between predict y (model's output) and true y
    loss_funcion = nn.CrossEntropyLoss()

    # optimizer is the function which will change the parameters of model according to loss, (learning rate = 0.001) controls how big each update step is.
    optimizer_function = torch.optim.Adam(f_model.parameters(), lr=1e-3)

    train(f_model, loss_funcion, optimizer_function, device)