import torch
from torch.utils.data import DataLoader, TensorDataset
from pipelines.atmospheric.gfs.ml.models.autoencoder import GFSForecastAutoencoder


def train_autoencoder(features, epochs=10, batch_size=32, lr=1e-3):
    print("Running")
    dataset = TensorDataset(features, features)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GFSForecastAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

    return model
