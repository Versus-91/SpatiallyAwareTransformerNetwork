import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split


class WSIDataset(Dataset):
    """
    Dataset class for WSI patch embeddings.

    Parameters:
        samples (List[Tensor]): List of [N_patches, D] tensors per WSI.
        labels (List[Tensor]): Corresponding labels for each WSI.
        coordinates (List[Tensor]): Coordinated of patches per WSI.
        n_patches (int): Number of patches to sample per WSI.
        pad_mode (str): Padding strategy if WSI has fewer patches.
    """

    def __init__(self, samples, labels, coordinates, n_patches=128, pad_mode='zero', seed=42):
        if not (len(samples) == len(labels) == len(coordinates)):
            raise ValueError(
                f"Inconsistent lengths: samples({len(samples)}), labels({len(labels)}), coordinates({len(coordinates)})"
            )
        self.samples = samples
        self.coordinates = coordinates
        self.labels = labels
        self.n_patches = n_patches
        self.pad_mode = pad_mode
        self.seed = seed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]  # shape: [num_patches, embed_dim]
        label = self.labels[idx]
        g = torch.Generator()
        g.manual_seed(self.seed + idx)
        coordinates = self.coordinates[idx]
        num_patches = sample.shape[0]

        if num_patches >= self.n_patches:
            indices = torch.randperm(num_patches, generator=g)[:self.n_patches]
            sampled_patches = sample[indices]
            sampled_coordinates = coordinates[indices]
            assert len(sampled_patches) == len(sampled_coordinates)

        else:
            if self.pad_mode == 'zero':
                padding = torch.zeros(
                    self.n_patches - num_patches, sample.shape[1], device=sample.device)
                padding_coordinate = torch.zeros(
                    self.n_patches - num_patches, coordinates.shape[1], device=sample.device)
            else:
                raise ValueError(f"{self.pad_mode} not implemented.")
            sampled_patches = torch.cat([sample, padding], dim=0)
            sampled_coordinates = torch.cat(
                [coordinates, padding_coordinate], dim=0)

        return sampled_patches, label, sampled_coordinates


if __name__ == '__main__':
    samples = [torch.randn(64, 3) for _ in range(10)]
    coords = [torch.randn(64, 2) for _ in range(10)]
    labels = [torch.randint(0, 2, (1,)) for _ in range(10)]
    dataset = WSIDataset(samples, labels, coords)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    for batch_patches, batch_labels, batch_coords in train_loader:
        print(f"Batch patches shape: {batch_patches.shape}")
        print(f"Batch coords shape: {batch_coords.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        break
