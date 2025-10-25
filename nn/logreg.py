import torch
import torch.nn as nn 

class LogisticRegression(nn.Module):
    """
    Plain logistic regression classifier.
    - If you pass tabular data shaped (N, D), it uses D features directly.
    - If you pass images shaped (N, C, H, W), it will flatten to (N, C*H*W).
    - Set num_classes=1 for binary (use BCEWithLogitsLoss).
    - Set num_classes>1 for multi-class (use CrossEntropyLoss).
    """
    def __init__(self, num_classes: int = 1):
        super().__init__()

        # define layer i: flatten features (no parameters)
        self.flatten = nn.Flatten(start_dim=1)

        # define layer ii: single linear map from features -> logits
        # LazyLinear infers the input feature count on first forward pass
        self.linear = nn.LazyLinear(out_features=num_classes, bias=True)

    def forward(self, x):
        # apply layer i (flatten)
        x1 = self.flatten(x)                      # (N, D) where D = product of non-batch dims
        # print(f'layer i (flatten): {x1.shape}')

        # apply layer ii (linear logits)
        logits = self.linear(x1)                  # (N, num_classes)
        # print(f'layer ii (logits): {logits.shape}')

        return logits


if __name__ == "__main__":
    # Select device: use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a random tensor simulating an input batch
    # Example: MNIST-like images (N, C, H, W) = (batch, 1, 28, 28)
    x = torch.randn(8, 1, 28, 28, device=device)  # batch of 8 random "images"

    # Create and move model to device
    model = LogisticRegression(num_classes=10).to(device)

    # Run a forward pass
    logits = model(x)

    # Print output details
    print("Input shape:", x.shape)
    print("Output (logits) shape:", logits.shape)
    print("Output sample:", logits[0])
