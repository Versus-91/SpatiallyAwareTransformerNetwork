# dataset/load_wsi_data.py

import os
import tarfile
import h5py
import torch
import pandas as pd


def extract_tar(archive_path, extract_path):
    with tarfile.open(archive_path, 'r') as archive:
        archive.extractall(path=extract_path)


def load_wsi_embeddings(h5_dir, labels_path, device='cpu'):
    features = []
    coordinates = []
    names = []
    sample_classes = []

    labels = pd.read_csv(labels_path, delimiter='\t')
    labels['SampleId'] = labels['studyID:sampleId'].apply(
        lambda x: x.split(":")[1])
    id_length = 15

    assert labels['SampleId'].is_unique and labels['SampleId'].str.len().eq(
        id_length).all()

    file_names = [f for f in os.listdir(h5_dir) if f.endswith('.h5')]

    for file_name in file_names:
        file_path = os.path.join(h5_dir, file_name)
        with h5py.File(file_path, 'r') as f:
            try:
                sample_id = file_name[:id_length]
                sample_class = labels.loc[labels['SampleId']
                                          == sample_id, 'CDH1'].iloc[0]
                assert not pd.isna(
                    sample_class), f"CDH1 value is NaN for SampleId {sample_id}"

                features.append(torch.tensor(f['feats'][:]).to(device))
                coordinates.append(torch.tensor(f['coords'][:]).to(device))
                names.append(sample_id)
                sample_classes.append(torch.tensor(
                    sample_class).float().to(device))
            except Exception as e:
                print(f"Error loading {file_name}: {e}")

    return names, features, coordinates, sample_classes
