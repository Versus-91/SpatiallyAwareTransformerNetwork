from dataset.load_wsi_data import load_wsi_embeddings
from dataset.wsi_dataset import WSIDataset
from torch.utils.data import DataLoader, random_split
from dataset.wsi_dataset import WSIDataset


if __name__ == '__main__':
    names, feats, coords, labels = load_wsi_embeddings(
        'temp', 'data/sample_matrix.txt', device='cuda')
    dataset = WSIDataset(feats, labels, coords, n_patches=32)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    for batch_patches, batch_labels, batch_coords in train_loader:
        print(f"Batch patches shape: {batch_patches.shape}")
        print(f"Batch coords shape: {batch_patches.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        break
