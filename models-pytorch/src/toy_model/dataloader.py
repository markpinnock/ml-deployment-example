import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class ToyLinearDataset(Dataset):  # type: ignore[misc]
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

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int) -> tuple[float, float]:
        return self.xs[idx], self.ys[idx]


def get_dataloaders(
    world_size: int, rank: int, local_batch_size: int
) -> dict[str, DataLoader | DistributedSampler | Dataset]:
    """Get dataloaders for training and validation datasets.

    Args:
        world_size: number of processes
        rank: process rank
        local_batch_size: batch size per process

    Returns:
        train_loader: training dataloader
        valid_loader: validation dataloader
        train_sampler: training sampler
        valid_sampler: validation sampler
        train_dataset: training dataset
        valid_dataset: validation dataset
    """
    train_dataset = ToyLinearDataset(train_split=True)
    valid_dataset = ToyLinearDataset(train_split=False)
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    valid_sampler = DistributedSampler(
        valid_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    train_loader = DataLoader(
        train_dataset, batch_size=local_batch_size, shuffle=False, sampler=train_sampler
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=local_batch_size, shuffle=False, sampler=valid_sampler
    )

    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "train_sampler": train_sampler,
        "valid_sampler": valid_sampler,
        "train_dataset": train_dataset,
        "valid_dataset": valid_dataset,
    }
