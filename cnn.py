from pathlib import Path
import torch
import time
import math
from data_loader import load_data

from visualization import (
    evaluate,
    plot_sample_grid,
    plot_learning_curves,
    plot_confusion_matrix
)

# ===================== Utility Functions ===================== #
def relu(x):
    return torch.maximum(x, torch.zeros_like(x))

def softmax(x):
    x = x - torch.max(x, dim=1, keepdim=True)[0]
    exp_x = torch.exp(x)
    return exp_x / torch.sum(exp_x, dim=1, keepdim=True)

# ===================== CNN Structure ===================== #
class CNN:
    def __init__(self, input_size, num_filters, kernel_size, fc_output_size, lr, device):
        self.device = device
        self.input_size = input_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.fc_output_size = fc_output_size
        self.lr = lr

        self.out_dim = input_size - kernel_size + 1
        self.flatten_size = num_filters * self.out_dim**2
        
        # self.kernel = np.random.randn(num_filters, 1, kernel_size, kernel_size) * 0.01
        self.kernel = torch.randn(num_filters, 1, kernel_size, kernel_size, device=device) * math.sqrt(2.0 / (kernel_size**2))
        
        # self.w2 = np.random.randn(fc_output_size, self.flatten_size) * 0.01
        self.w2 = torch.randn(fc_output_size, self.flatten_size, device=device) * math.sqrt(2.0 / self.flatten_size)
        self.b2 = torch.zeros((fc_output_size, 1), device=device) * 0.01

        self.x_input = None
        self.z1 = None
        self.za = None
        self.za2 = None
        self.a1_flat = None

    def forward(self, x):
        """ Forward propagation """
        batch_size = x.shape[0]
        self.x_input = x
        batch_size, channel, H, W = x.shape

        z1 = torch.zeros(batch_size, self.num_filters, self.out_dim, self.out_dim, device=self.device)
        # for b in range(batch_size):
        #     for f in range(self.num_filters):
        #         for h in range(self.out_dim):
        #             for w in range(self.out_dim):
        #                 # representing the slice where the kernel is, over each images 
        #                 slice_matrix = x[b, 0, h: h+self.kernel_size, w: w+self.kernel_size]
        #                 # sum of the slice * the kernel's weight
        #                 # [f,0]: kernel is 4d, we specified the filter and channel 
        #                 z1[b,f,h,w] = np.sum(slice_matrix * self.kernel[f,0])
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                slice_matrix = x[:, 0, i:i+self.out_dim, j:j+self.out_dim]

                kernel_weight = self.kernel[:,0,i,j]

                z1 += slice_matrix.unsqueeze(1) * kernel_weight.view(1, -1, 1, 1)


        self.z1 = z1
        a1 = relu(z1)
        self.a1 = a1
        # Flatten each image's convolutional feature maps.
        a1_flat = a1.reshape(batch_size, self.flatten_size) 
        self.a1_flat = a1_flat
        z2 = torch.mm(a1_flat, self.w2.T) + self.b2.T
        self.z2 = z2
        outputs = softmax(z2) #y_hat

        return outputs

    def backward(self, y, pred):
        """ Backward propagation """
        # 1. one-hot encode the labels
        batch_size = y.shape[0]
        y_one_hot = torch.zeros(batch_size, self.fc_output_size, device=self.device)
        y_one_hot[torch.arange(batch_size, device=self.device), y] = 1

        # 2. Calculate softmax cross-entropy loss gradient
        dz2 = pred - y_one_hot
        dw2 = dz2.T @ self.a1_flat / batch_size
        db2 = torch.sum(dz2, dim=0, keepdim=True).T / batch_size
        # 3. Calculate fully connected layer gradient
        dza_flat = dz2 @ self.w2

        dza = dza_flat.reshape(batch_size, self.num_filters, self.out_dim, self.out_dim)
        
        # 4. Backpropagate through ReLU
        dz1 = dza * (self.z1 > 0)
        # 5. Calculate convolution kernel gradient
        dw1 = torch.zeros_like(self.kernel, device=self.device)
        # for b in range(batch_size):
        #     for f in range(self.num_filters):
        #         for h in range(self.out_dim):
        #             for w in range(self.out_dim):
        #                 slice_matrix = self.x_input[b, 0, h: h+self.kernel_size, w: w+self.kernel_size]
        #                 dw1[f,0] += slice_matrix * dz1[b,f,h,w]
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                slice_matrix = self.x_input[:, 0, i:i+self.out_dim, j:j+self.out_dim]
                dw1[:,0,i,j] = torch.sum(slice_matrix.unsqueeze(1) * dz1, dim=(0,2,3))
        dw1 = dw1 / batch_size
        # 6. Update parameters
        self.kernel -= self.lr * dw1
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        
    @torch.no_grad()
    def train(self, x, y):
        # call forward function
        outputs = self.forward(x)
        # calculate loss
        batch_size = outputs.shape[0]
        correct_class_probs = outputs[torch.arange(batch_size, device=self.device), y]
        loss = -torch.mean(torch.log(correct_class_probs.clamp(min=1e-10)))

        # call backward function
        self.backward(y, outputs)
        return loss
# ===================== Training Process ===================== #
def main():
    output_dir = Path(__file__).resolve().parent / "result" / "cnn"
    output_dir.mkdir(parents=True, exist_ok=True)

    # First, load data
    train_loader, val_loader, test_loader = load_data(batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(f"Using device: {device}")
    if device.type == "cuda": print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    

    # Second, define hyperparameters
    input_size = 28
    num_epochs = 150
    patience = 10
    num_filters = 8
    kernel_size = 3
    fc_output_size = 10
    lr = 0.099
    
    model = CNN(input_size, num_filters, kernel_size, fc_output_size, lr, device)

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
    parameter_names = ('kernel', 'w2', 'b2')
    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in train_loader:  # define training phase for training model
            x = inputs.to(device)
            y = labels.to(device)
            loss = model.train(x, y)
            total_loss += loss.detach() * labels.shape[0]

        avg_epoch_loss = (total_loss / len(train_loader.dataset)).item()
        epoch_losses.append(avg_epoch_loss)
        train_loss, train_acc = evaluate(
            model, train_loader, device, flatten=False
        )
        val_loss, val_acc = evaluate(
            model, val_loader, device, flatten=False
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
        flatten=False,
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
        "CNN",
        output_dir / "cnn_learning_curves.png",
    )
    plot_confusion_matrix(
        y_true,
        y_pred,
        "CNN",
        output_dir / "cnn_confusion_matrix.png",
    )

if __name__ == "__main__":
    main()
