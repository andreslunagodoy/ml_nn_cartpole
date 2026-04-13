import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils.logger import logger, CSVLogger
from config import STATE_DIM


def train(model, X_train, delta_train, r_train,
          X_val, delta_val, r_val,
          epochs=20, batch_size=64, reward_weight=0.1, verbose=True):

    train_dataset = TensorDataset(X_train, delta_train, r_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.MSELoss(reduction='mean')
    csv_logger = CSVLogger("wm_training", ["epoch", "train_loss", "val_loss", "val_state_mse", "val_reward_mse"])

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = 5

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for x, d, r in train_loader:
            pred = model(x)
            pred_delta, pred_reward = pred[:, :STATE_DIM], pred[:, STATE_DIM:]

            loss_d = criterion(pred_delta, d)
            loss_r = criterion(pred_reward, r.unsqueeze(1))
            loss = loss_d + reward_weight * loss_r

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accumulate properly (sum over samples)
            batch_size_actual = x.size(0)
            total_loss += loss.item() * batch_size_actual
            total_samples += batch_size_actual

        # average over all samples
        train_loss = total_loss / total_samples

        val_metrics = evaluate(model, X_val, delta_val, r_val, reward_weight)

        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        scheduler.step(val_metrics['total'])

        csv_logger.log(epoch + 1, train_loss, val_metrics['total'], val_metrics['state_mse'], val_metrics['reward_mse'])

        if verbose:
            logger.info(f"WM Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_metrics['total']:.6f}")

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    csv_logger.close()


def evaluate(model, X, delta, r, reward_weight=0.1):
    model.eval()
    with torch.no_grad():
        pred = model(X)
        pred_delta, pred_reward = pred[:, :STATE_DIM], pred[:, STATE_DIM:]

        loss_d = ((pred_delta - delta) ** 2).mean()
        loss_r = ((pred_reward - r.unsqueeze(1)) ** 2).mean()

        loss = loss_d + reward_weight * loss_r

    return {
        'total': loss.item(),
        'state_mse': loss_d.item(),
        'reward_mse': loss_r.item(),
    }