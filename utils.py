import torch


def train(model, dataloader, optimizer, criterion, epochs=10, ignore_coordinates=True):
    model.train()
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for patches, label, coords in dataloader:
            optimizer.zero_grad()
            label = label.float()
            if ignore_coordinates:
                logits = model(patches)
            else:
                logits = model(patches, coords)
            loss = criterion(logits.squeeze(-1), label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(dataloader):.4f}")
    return losses


@torch.no_grad()
def predict(model, data_loader, device):
    model.eval()  # Set model to evaluation mode
    predictions = []
    true_labels = []
    probas = []
    for patches, labels, coords in data_loader:

        patches = patches.to(device)
        labels = labels.to(device)

        outputs = model(patches)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        predictions.append(preds.cpu())
        true_labels.append(labels.cpu())
        probas.append(probs.cpu())

    predictions = torch.cat(predictions, dim=0)
    true_labels = torch.cat(true_labels, dim=0)
    probas = torch.cat(probas, dim=0)

    return predictions, true_labels, probas
