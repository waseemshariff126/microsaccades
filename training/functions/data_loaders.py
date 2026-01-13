import torch
import torchvision
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import warnings
import os
import numpy as np
from os.path import isfile, join

warnings.filterwarnings('ignore')


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h = img.size(2)
        w = img.size(3)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img


class microsaccade(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize = transforms.Resize(size=(440, 300), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        self.rotate = transforms.RandomRotation(degrees=30)
        self.shearx = transforms.RandomAffine(degrees=0, shear=(-30, 30))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        sample = np.load(f"{self.root}/{index}.npy", allow_pickle=True)

        data = sample.item()["data"]  # Shape: [N, 4] -> [T, X, Y, P]
        target = sample.item()["target"]

        # Convert time to milliseconds
        data[:, 0] *= 1000  # Convert T from seconds to milliseconds

        # Convert to NumPy before processing
        if isinstance(data, torch.Tensor):  
            data = data.cpu().numpy()

        # Round values and convert to int
        data = np.round(data).astype(np.int32)

        # Clamp X and Y to be within valid range
        data[:, 1] = np.clip(data[:, 1], 0, 799)  # X (0-799)
        data[:, 2] = np.clip(data[:, 2], 0, 599)  # Y (0-599)

        # Center crop on X: From 180 to 620 on the X axis (center 440 pixels)
        center_x_start = 180
        center_x_end = 620
        data = data[(data[:, 1] >= center_x_start) & (data[:, 1] < center_x_end)]  # Crop X

        # Top crop on Y: From 0 to 300 on the Y axis
        data[:, 2] = np.clip(data[:, 2], 0, 300)  # Crop Y to top 300 pixels

        # Rename events according to the new resolution (440x300)
        # New X should range from 0 to 439
        data[:, 1] = data[:, 1] - center_x_start  # Shift X to start from 0 (center cropped)
        
        # New Y should range from 0 to 299
        # Y is already clamped to be from 0 to 300, so it's ready
        # We don't need additional normalization for Y as it directly fits in 300px range

        # Now integrate the events into frames with the new cropped resolution (300x440)
        new_frames = integrate_events_to_frames(data[:, 1], data[:, 2], data[:, 3], 300, 440, frames_number=10)

        target = torch.tensor(target, dtype=torch.long)
        
        new_data = torch.tensor(new_frames, dtype=torch.int32)

        # Apply resizing after cropping (optional step, based on your requirements)
        new_data = self.resize(new_data)

        # Cast new_data to float32
        new_data = new_data.float()  # Convert to float32

        # Apply augmentation if needed
        if self.transform:
            choices = ['roll', 'rotate', 'shear']
            aug = np.random.choice(choices)
            if aug == 'roll':
                off1 = random.randint(-5, 5)
                off2 = random.randint(-5, 5)
                new_data = torch.roll(new_data, shifts=(off1, off2), dims=(2, 3))
            if aug == 'rotate':
                new_data = self.rotate(new_data)
            if aug == 'shear':
                new_data = self.shearx(new_data)

        return new_data, target.long().squeeze(-1)


    def __len__(self):
        return len(os.listdir(self.root))



def integrate_events_to_frames(x, y, p, H, W, frames_number):
    frame_interval = len(x) // frames_number
    crop_width = 440
    crop_height = 300

    x_min = 180
    x_max = 620
    y_min = 0
    y_max = crop_height

    frames = np.zeros((frames_number, 2, crop_height, crop_width), dtype=np.float32)

    for f in range(frames_number):
        j_l = f * frame_interval
        j_r = (f + 1) * frame_interval if f < frames_number - 1 else len(x)

        x_seg, y_seg, p_seg = x[j_l:j_r], y[j_l:j_r], p[j_l:j_r]

        # Filter events based on the defined central top region
        mask = (x_seg >= x_min) & (x_seg < x_max) & (y_seg >= y_min) & (y_seg < y_max)
        x_seg, y_seg, p_seg = x_seg[mask], y_seg[mask], p_seg[mask]

        # Adjust coordinates for cropping
        x_seg -= x_min
        y_seg -= y_min

        for c in range(2):
            mask = (p_seg == c)
            pos = (y_seg[mask] * crop_width + x_seg[mask]).astype(np.int64)
            frame = np.bincount(pos, minlength=crop_height * crop_width).reshape(crop_height, crop_width)
            frames[f, c] = frame

    return frames
 

def build_microsaccade(path='us_dataset/left-resampled', transform=False):
    train_path = path + '/train'
    val_path = path + '/test'
    train_dataset = microsaccade(root=train_path, transform=transform)
    val_dataset = microsaccade(root=val_path, transform=False)

    return train_dataset, val_dataset


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def rand_bbox(size, lam):
    W = size[3]
    H = size[4]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(input, target, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(input.size()[0]).cuda()

    target_a = target
    target_b = target[rand_index]

    # generate mixed sample
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    input[:, :, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
    return input, target_a, target_b, lam


if __name__ == '__main__':
    choices = ['roll', 'rotate', 'shear']
    aug = np.random.choice(choices)
    print(aug) 
    