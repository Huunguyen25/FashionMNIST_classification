from pathlib import Path
from binascii import a2b_base64
from math import log
from turtle import forward
import numpy as np
import time
import torch

from data_loader import load_data
from visualization import (
    evaluate,
    plot_sample_grid,
    plot_learning_curves,
    plot_confusion_matrix,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sigmoid(x):  # manually define the sigmoid
    return 1/(1 + torch.exp(-x))

def softmax(x):  # define the softmax 
    x = x - torch.max(x, axis=1, keepdims=True).values
    exp_x = torch.exp(x)
    return exp_x / torch.sum(exp_x, axis=1, keepdims=True)

class MLP:
    def __init__(self, input_size, hidden_size, a, lr):  # 
        
        self.lr = lr
        
        self.Weight1 = torch.randn(hidden_size, input_size, device=device) * 0.01
        self.bias1 = torch.zeros((hidden_size, 1), device=device)
        self.Weight2 = torch.randn(a, hidden_size, device=device) * 0.01
        self.bias2 = torch.zeros((a, 1), device=device)    
        
        self.z1 = None
        self.a1 = None
        self.z2 = None
    
    def forward(self, x):  # forward propagation to get predictions
        # z = W * a + b
        # a = g(z)
        z1 = (x @ self.Weight1.T) + self.bias1.T # self.bias1 * 1^T: in lecture multiply by transposed 1 matrix
        a1 = sigmoid(z1)
        
        z2 = (a1 @ self.Weight2.T) + self.bias2.T
        outputs = softmax(z2) 
        
        self.z1 = z1
        self.a1 = a1 
        self.z2 = z2
        
        # 10 by 128 vector
        # rows of class guesses x 128 images per batch
        return outputs
    
    # pred = output from forward 
    # y = true labels
    # x = input (128 by 784 matrix)
    def backward(self, x, y, pred):
        batch_size = x.shape[0]
        # one-hot encode the labels

        # compute the gradients
        
        # CE + softmax loss function derivation result in y^ - y
        # dZ2 is guess - true class
        dz2 = pred - y
        #updated weight of second layer
        #dW^(l) = dZ^(l) * A^(l-1)^T
        dw2 = (dz2.T @ self.a1) / batch_size
        #updated bias of second layer
        db2 = torch.sum(dz2, axis=0, keepdims=True).T / batch_size
        #dA^(l-1) = W^(l) * dZ^(l)
        dA1 = dz2 @ self.Weight2

        dz1 = (self.a1 * (1-self.a1)) * dA1
        dw1 = (dz1.T @ x) / batch_size
        db1 = torch.sum(dz1, axis=0, keepdims=True).T / batch_size
        
        # update the weights and biases
        self.Weight2 -= self.lr * dw2
        self.bias2 -= self.lr *db2
        
        self.Weight1 -= self.lr * dw1
        self.bias1 -= self.lr * db1
        

    def train(self, x,y):
        # call forward function
        output = self.forward(x)
        # calculate loss,
        # 1e-12 avoid log(0)
        epsilon = 1e-12
        output = torch.clamp(output, epsilon, 1-epsilon)
        loss = torch.mean(-torch.sum(y*torch.log(output), axis=1))
        # call backward function
        self.backward(x, y, output)
        
        return loss

def main():
    output_dir = Path(__file__).resolve().parent / "result" / "mlp"
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 64
    # First, load data
    train_loader, val_loader, test_loader = load_data(batch_size)
    images, labels = next(iter(train_loader))

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Pixel range:", images.min().item(), images.max().item())
    print("First 10 labels:", labels[:10])

    # Second, define hyperparameters
    input_size = 28*28  # MNIST images are 28x28 pixels
    num_epochs = 150
    patience = 10
    hidden_layer_size = 192
    num_classes = 10
    learning_rate = 0.09943822786021822

    # model object here
    model = MLP(input_size=input_size, hidden_size=hidden_layer_size, a=num_classes, lr=learning_rate)

    # Then, train the model
    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.time()
    epoch_losses = []
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Keep an independent copy of the weights with the lowest validation loss.
    parameter_names = ('Weight1', 'bias1', 'Weight2', 'bias2')
    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(num_epochs):
        total_loss = 0

        for inputs, labels in train_loader:  # define training phase for training model
            batch_size = inputs.shape[0]

            x = inputs.view(batch_size, input_size).to(device)
            y_labels = labels.to(device)

            y = torch.zeros(batch_size, num_classes, device=device)
            y.scatter_(1, y_labels.unsqueeze(1), 1.0)

            loss = model.train(x,y)
            total_loss += loss.detach() * labels.shape[0]

        avg_epoch_loss = (total_loss / len(train_loader.dataset)).item()
        epoch_losses.append(avg_epoch_loss)
        train_loss, train_acc = evaluate(
            model, train_loader, device, flatten=True
        )
        val_loss, val_acc = evaluate(
            model, val_loader, device, flatten=True
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(
            f"Epoch {epoch+1}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Train Accuracy: {train_acc:.4f}, "
            f"Val Accuracy: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_weights = {
                name: getattr(model, name).detach().clone()
                for name in parameter_names
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch + 1} epochs.")
                break


    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total training time: {training_time:.2f} seconds")
 
    # Finally, evaluate the model

    if device.type == "cuda":
        torch.cuda.synchronize()
    if best_weights is None:
        raise RuntimeError("No finite validation loss; cannot select a best model.")
    for name, weights in best_weights.items():
        setattr(model, name, weights)
    print(f"Restored epoch {best_epoch} (validation loss: {best_val_loss:.4f}).")

    eval_start_time = time.time()

    test_loss, test_acc, y_true, y_pred = evaluate(
        model,
        test_loader,
        device,
        flatten=True,
        collect_predictions=True,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    eval_end_time = time.time()
    eval_time = eval_end_time - eval_start_time
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Total evaluation time: {eval_time:.2f} seconds")
    
    total_elapsed_time = training_time + eval_time
    print(f"Total time (training + evaluation): {total_elapsed_time:.2f} seconds")

    plot_sample_grid(
        train_loader.dataset,
        save_path=output_dir / "fashionmnist_samples.png",
    )
    plot_learning_curves(
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
        "MLP",
        output_dir / "mlp_learning_curves.png",
    )
    plot_confusion_matrix(
        y_true,
        y_pred,
        "MLP",
        output_dir / "mlp_confusion_matrix.png",
    )


if __name__ == "__main__":
    main()  