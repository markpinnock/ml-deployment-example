"""Based on: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html"""
# https://github.com/Azure/optimized-pytorch-on-databricks-and-fabric/blob/main/Azure%20Databricks/model_training_fsdp.ipynb
# https://pytorch.org/docs/stable/distributed.html
# https://stackoverflow.com/questions/66226135/how-to-parallelize-a-training-loop-ever-samples-of-a-batch-when-cpu-is-only-avai
# https://pytorch.org/docs/stable/multiprocessing.html#module-torch.multiprocessing.spawn

import os
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU to mimic multiple GPUs

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset


def setup(rank: int, world_size: int) -> None:
    """Initialize distributed environment.

    Args:
        rank: device number
        world_size: number of devices
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12335"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup() -> None:
    """Tear down distributed environment."""
    dist.destroy_process_group()


class LinearNet(nn.Module):
    """Toy MLP, trainable on CPU."""

    def __init__(self, layer_dims: list[int]) -> None:
        """Initialise LinearNet.

        Args:
            layer_dims: dimensions for each layer including input and output
        """
        super().__init__()
        layers = []

        for idx in range(len(layer_dims) - 2):
            layers.append(
                nn.Linear(in_features=layer_dims[idx], out_features=layer_dims[idx + 1])
            )
            layers.append(nn.ReLU())

        layers.append(
            nn.Linear(in_features=layer_dims[-2], out_features=layer_dims[-1])
        )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass."""
        for layer in self.layers:
            x = layer(x)
        return x


def train(
    model: nn.Module,
    train_loader,
    loss_obj: nn.MSELoss,
    optimiser: torch.optim.Optimizer,
    rank: int,
) -> float:
    """Train for one epoch.

    Args:
        model: model to be trained
        train_loader: dataloader
        loss_obj: loss function
        optimiser: optimiser
        rank: device number
    Returns:
        epoch training loss
    """
    model.train()
    ddp_loss = torch.zeros(2)

    for data, labels in train_loader:
        # Forward pass
        data = data.to(rank) if torch.cuda.is_available() else data
        labels = labels.to(rank) if torch.cuda.is_available() else labels
        optimiser.zero_grad()
        preds = model(data)

        # Calculate loss and backpropagate
        loss = loss_obj(preds, labels)
        loss.backward()
        optimiser.step()

        # Add to running total
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data)

    return (torch.sum(ddp_loss[0]) / torch.sum(ddp_loss[1])).item()


def test(model: nn.Module, valid_loader, loss_obj: nn.MSELoss, rank: int) -> float:
    """Test after training for one epoch.

    Args:
        model: model to be evaluated
        valid_loader: dataloader
        loss_obj: loss function
        rank: device number

    Returns:
        epoch validation loss
    """
    model.eval()
    ddp_loss = torch.zeros(2)

    with torch.inference_mode():
        for data, labels in valid_loader:
            # Forward pass and calculate loss
            data = data.to(rank) if torch.cuda.is_available() else data
            labels = labels.to(rank) if torch.cuda.is_available() else labels
            preds = model(data)
            loss = loss_obj(preds, labels)

            # Add to running total
            ddp_loss[0] += loss.item()
            ddp_loss[1] += len(data)

    return (torch.sum(ddp_loss[0]) / torch.sum(ddp_loss[1])).item()


class ToyLinearDataset(Dataset):
    """Toy example dataset - linear relationship"""

    def __init__(self, train_split: bool):
        """Initialise linear dataset.

        Args:
            train_split: training dataset if True (else validation)
        """
        super().__init__()

        alpha = 0.2
        beta = 0.4
        sigma = 0.02

        self.xs = (
            torch.linspace(0.0, 1.0, 50)
            if train_split
            else torch.linspace(1.0, 1.2, 10)
        )
        eps = torch.randn_like(self.xs) * sigma
        self.ys = alpha + beta * self.xs + eps
        self.xs = self.xs.unsqueeze(1)
        self.ys = self.ys.unsqueeze(1)
        self.xs = torch.concat([torch.ones_like(self.xs), self.xs], dim=1)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx: int) -> tuple[float, float]:
        return self.xs[idx], self.ys[idx]


def main(rank: int, world_size: int) -> None:
    """Train model."""
    checkpoint_path = tempfile.gettempdir() + "/checkpoint.pth"

    setup(rank, world_size)

    epochs = 100
    save_every = 10

    train_dataset = ToyLinearDataset(train_split=True)
    valid_dataset = ToyLinearDataset(train_split=False)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
    print(rank, False, next(iter(valid_loader))[0][:, 1])

    if torch.cuda.is_available():
        model = LinearNet([2, 1]).to(rank)
        ddp_model = DistributedDataParallel(model, device_ids=[rank])
    else:
        model = LinearNet([2, 1]).share_memory()
        ddp_model = DistributedDataParallel(model)

    # Save model to checkpoint and synchronise to avoid loading prematurely
    if rank == 0:
        torch.save(ddp_model.state_dict(), checkpoint_path)
    dist.barrier()

    # Load model to all devices from checkpoint
    if torch.cuda.is_available():
        map_location = {"cuda:0": f"cuda:{rank}"}
    else:
        map_location = "cpu"
    ddp_model.load_state_dict(
        torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    )

    # Set up optimiser and loss function
    optimiser = torch.optim.Adam(ddp_model.parameters(), lr=0.01)
    loss_obj = nn.MSELoss(reduction="sum")

    train_losses, valid_losses = [], []

    for epoch in range(1, epochs + 1):
        train_loss = train(ddp_model, train_loader, loss_obj, optimiser, rank)
        valid_loss = test(ddp_model, valid_loader, loss_obj, rank)

        if rank == 0:
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            print(f"Epoch {epoch}: train {train_losses[-1]}, valid {valid_losses[-1]}")

            if epoch % save_every == 0:
                torch.save(ddp_model.state_dict(), checkpoint_path)

    # Plot results
    if rank == 0:
        true_train_ys = 0.2 + 0.4 * train_dataset.xs[:, 1]
        true_valid_ys = 0.2 + 0.4 * valid_dataset.xs[:, 1]

        with torch.inference_mode():
            train_data = (
                train_dataset.xs.to(0)
                if torch.cuda.is_available()
                else train_dataset.xs
            )
            valid_data = (
                valid_dataset.xs.to(0)
                if torch.cuda.is_available()
                else valid_dataset.xs
            )
            pred_train_ys = ddp_model(train_data).squeeze()
            pred_valid_ys = ddp_model(valid_data).squeeze()

        plt.subplot(2, 1, 1)
        plt.plot(np.linspace(1, epochs + 1, epochs), train_losses, label="Training")
        plt.plot(np.linspace(1, epochs + 1, epochs), valid_losses, label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(train_dataset.xs[:, 1], true_train_ys, c="k", label="Train")
        plt.scatter(train_dataset.xs[:, 1], pred_train_ys, c="k")
        plt.plot(valid_dataset.xs[:, 1], true_valid_ys, c="r", label="Valid")
        plt.scatter(valid_dataset.xs[:, 1], pred_valid_ys, c="r")
        plt.show()

    if rank == 0:
        os.remove(checkpoint_path)

    cleanup()


if __name__ == "__main__":
    world_size = 8  # Number of CPU cores to use
    mp.spawn(main, args=(world_size,), nprocs=world_size)
