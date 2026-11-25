
import numpy as np
import matplotlib.pyplot as plt
import torch

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


# ===================== Evaluation ===================== #
@torch.no_grad()
def evaluate(model, loader, device, flatten=False,
             collect_predictions=False):

    total_loss = 0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    for inputs, labels in loader:

        x = inputs.to(device)

        if flatten:
            x = x.reshape(x.shape[0], -1)

        y = labels.to(device)

        outputs = model.forward(x)

        correct_probs = outputs[
            torch.arange(y.shape[0], device=device), y
        ]

        loss = -torch.log(
            correct_probs.clamp_min(1e-10)
        ).mean()

        predicted = torch.argmax(outputs, dim=1)

        total_loss += loss.item() * y.shape[0]
        correct += (predicted == y).sum().item()
        total += y.shape[0]

        if collect_predictions:
            all_labels.append(y.cpu())
            all_predictions.append(predicted.cpu())

    avg_loss = total_loss / total
    accuracy = correct / total

    if collect_predictions:
        return (
            avg_loss,
            accuracy,
            torch.cat(all_labels).numpy(),
            torch.cat(all_predictions).numpy()
        )

    return avg_loss, accuracy


# ===================== Sample Image Grid ===================== #
def plot_sample_grid(dataset, save_path="fashionmnist_samples.png"):

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(dataset))

    samples = [[] for _ in range(10)]

    for idx in indices:
        image, label = dataset[int(idx)]
        label = int(label)

        if len(samples[label]) < 10:
            samples[label].append(image)

        if all(len(row) == 10 for row in samples):
            break

    fig, axes = plt.subplots(10, 10, figsize=(12, 12))

    for row in range(10):
        for col in range(10):

            image = samples[row][col]

            axes[row, col].imshow(
                image.squeeze().numpy(),
                cmap="gray"
            )

            axes[row, col].axis("off")

        axes[row, 0].set_ylabel(
            CLASS_NAMES[row],
            rotation=0,
            labelpad=48,
            va="center"
        )

    fig.suptitle("Fashion-MNIST Sample Images", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ===================== Learning Curves ===================== #
def plot_learning_curves(
    train_losses,
    val_losses,
    train_accuracies,
    val_accuracies,
    model_name,
    save_path
):

    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, train_losses, label="Training")
    axes[0].plot(epochs, val_losses, label="Validation")

    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(
        epochs,
        np.array(train_accuracies) * 100,
        label="Training"
    )

    axes[1].plot(
        epochs,
        np.array(val_accuracies) * 100,
        label="Validation"
    )

    axes[1].set_title("Training and Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True)

    fig.suptitle(f"{model_name} Learning Curves", fontsize=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ===================== Confusion Matrix ===================== #
def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name,
    save_path
):

    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(cm, cmap="Blues")

    ax.set_title(f"{model_name} Test Set Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))

    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)

    threshold = cm.max() / 2

    for i in range(10):
        for j in range(10):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=8
            )

    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
