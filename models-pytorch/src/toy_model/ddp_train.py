"""Based on: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html"""

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
from torch.utils.data import DataLoader
from toy_model.dataloader import get_dataloaders
from toy_model.model import LinearNet

torch.manual_seed(5)


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


def train(
    model: nn.Module,
    train_loader: DataLoader,
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
    ddp_loss = torch.zeros(2).to(rank) if torch.cuda.is_available() else torch.zeros(2)

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

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    return float((ddp_loss[0] / ddp_loss[1]).item())


def test(
    model: nn.Module, valid_loader: DataLoader, loss_obj: nn.MSELoss, rank: int
) -> float:
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
    ddp_loss = torch.zeros(2).to(rank) if torch.cuda.is_available() else torch.zeros(2)

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

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    return float((ddp_loss[0] / ddp_loss[1]).item())


def main(rank: int, world_size: int) -> None:
    """Train model."""
    checkpoint_path = tempfile.gettempdir() + "/checkpoint.pth"

    setup(rank, world_size)

    epochs = 50
    save_every = 10
    learning_rate = 0.01  # * world_size
    global_batch_size = 8
    local_batch_size = global_batch_size // world_size

    # Set up dataloaders
    data_dict = get_dataloaders(world_size, rank, local_batch_size)

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
        map_location = {"cpu": "cpu"}
    ddp_model.load_state_dict(
        torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    )

    # Set up optimiser and loss function
    optimiser = torch.optim.Adam(ddp_model.parameters(), lr=learning_rate)
    loss_obj = nn.MSELoss(reduction="sum")

    train_losses, valid_losses = [], []

    for epoch in range(1, epochs + 1):
        data_dict["train_sampler"].set_epoch(epoch - 1)
        data_dict["valid_sampler"].set_epoch(epoch - 1)
        train_loss = train(
            ddp_model, data_dict["train_loader"], loss_obj, optimiser, rank
        )
        valid_loss = test(ddp_model, data_dict["valid_loader"], loss_obj, rank)

        # Print results if main process
        if rank == 0:
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            print(f"Epoch {epoch}: train {train_losses[-1]}, valid {valid_losses[-1]}")

            if epoch % save_every == 0:
                torch.save(ddp_model.state_dict(), checkpoint_path)

    # Plot results if main process
    if rank == 0:
        true_train_ys = 0.2 + 0.4 * data_dict["train_dataset"].xs[:, 1]
        true_valid_ys = 0.2 + 0.4 * data_dict["valid_dataset"].xs[:, 1]

        with torch.inference_mode():
            train_data = (
                data_dict["train_dataset"].xs.to(0)
                if torch.cuda.is_available()
                else data_dict["train_dataset"].xs
            )
            valid_data = (
                data_dict["valid_dataset"].xs.to(0)
                if torch.cuda.is_available()
                else data_dict["valid_dataset"].xs
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
        plt.plot(
            data_dict["train_dataset"].xs[:, 1], true_train_ys, c="k", label="Train"
        )
        plt.scatter(data_dict["train_dataset"].xs[:, 1], pred_train_ys, c="k")
        plt.plot(
            data_dict["valid_dataset"].xs[:, 1], true_valid_ys, c="r", label="Valid"
        )
        plt.scatter(data_dict["valid_dataset"].xs[:, 1], pred_valid_ys, c="r")
        plt.show()

    if rank == 0:
        os.remove(checkpoint_path)

    cleanup()


if __name__ == "__main__":
    world_size = 4  # Number of CPU cores to use
    mp.spawn(main, args=(world_size,), nprocs=world_size)
