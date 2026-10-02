import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import UNet
from dataset import get_segmentation_dataloaders


class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice Loss for robust semantic segmentation."""
    def __init__(self, smooth: float = 1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(inputs, targets)
        
        probs = torch.sigmoid(inputs)
        inputs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            inputs_flat.sum() + targets_flat.sum() + self.smooth
        )

        return bce_loss + dice_loss


@torch.no_grad()
def calculate_metrics(inputs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    probs = torch.sigmoid(inputs)
    preds = (probs > threshold).float()

    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum().item()
    total_union = (preds_flat + targets_flat).sum().item()

    dice = (2.0 * intersection + 1e-6) / (total_union + 1e-6)
    iou = (intersection + 1e-6) / (total_union - intersection + 1e-6)

    return dice, iou


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        dice, iou = calculate_metrics(outputs, masks)
        total_dice += dice * images.size(0)
        total_iou += iou * images.size(0)

        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    n = len(dataloader.dataset)
    return running_loss / n, total_dice / n, total_iou / n


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)

        running_loss += loss.item() * images.size(0)
        dice, iou = calculate_metrics(outputs, masks)
        total_dice += dice * images.size(0)
        total_iou += iou * images.size(0)

    n = len(dataloader.dataset)
    return running_loss / n, total_dice / n, total_iou / n


def main():
    parser = argparse.ArgumentParser(description="Train U-Net from scratch on synthetic segmentation")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=128)
    args = parser.parse_args()

    # Hardware detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load Data
    train_loader, val_loader = get_segmentation_dataloaders(
        batch_size=args.batch_size, img_size=args.img_size
    )

    # Initialize Model & Optimizer
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Starting U-Net training ({args.epochs} epochs)...")
    best_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} - Dice: {train_dice:.4f} - IoU: {train_iou:.4f} | "
            f"Val Loss: {val_loss:.4f} - Val Dice: {val_dice:.4f} - Val IoU: {val_iou:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "best_unet.pth")

    print(f"\nTraining Complete! Best Validation Dice Score: {best_dice:.4f}")


if __name__ == "__main__":
    main()