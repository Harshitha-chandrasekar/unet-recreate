import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SyntheticSegmentationDataset(Dataset):
    """Generates synthetic images with random circles and rectangles for binary mask segmentation."""
    def __init__(self, num_samples: int = 500, img_size: int = 128):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        np.random.seed(idx)
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        num_shapes = np.random.randint(1, 4)
        for _ in range(num_shapes):
            shape_type = np.random.choice(["circle", "rectangle"])
            color = np.random.uniform(0.3, 1.0, size=3)

            if shape_type == "circle":
                cx, cy = np.random.randint(20, self.img_size - 20, size=2)
                r = np.random.randint(10, 25)
                y, x = np.ogrid[:self.img_size, :self.img_size]
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                shape_mask = dist <= r
            else:
                x1, y1 = np.random.randint(10, self.img_size - 40, size=2)
                w, h = np.random.randint(15, 30, size=2)
                shape_mask = np.zeros((self.img_size, self.img_size), dtype=bool)
                shape_mask[y1:y1 + h, x1:x1 + w] = True

            img[shape_mask] = color
            mask[shape_mask] = 1.0

        # Add light background noise
        noise = np.random.normal(0, 0.05, img.shape).astype(np.float32)
        img = np.clip(img + noise, 0.0, 1.0)

        # Convert to PyTorch tensors (CHW layout)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return img_tensor, mask_tensor


def get_segmentation_dataloaders(batch_size: int = 16, img_size: int = 128):
    train_dataset = SyntheticSegmentationDataset(num_samples=800, img_size=img_size)
    val_dataset = SyntheticSegmentationDataset(num_samples=200, img_size=img_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader