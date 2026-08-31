import os
import os.path as osp
import sys
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from contextlib import contextmanager

def create_dir(dir_path):
    if not osp.exists(dir_path):
        print('Path {} does not exist. Creating it...'.format(dir_path))
        os.makedirs(dir_path)

def parse_defense_kwargs(kwargs_str):
    kwargs = dict()
    for entry in kwargs_str.split(';'):
        if len(entry) < 1:
            continue
        key, value = entry.split(':')
        assert key not in kwargs, 'Argument ({}:{}) conflicts with ({}:{})'.format(key, value, key, kwargs[key])
        try:
            # Cast into int if possible
            value = int(value)
        except ValueError:
            try:
                # Try with float
                value = float(value)
            except ValueError:
                # Give up
                pass
        kwargs[key] = value
    return kwargs

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


class TransferSetImagePaths(ImageFolder):
    """TransferSet Dataset, for when images are stored as *paths*"""

    def __init__(self, samples, transform=None, target_transform=None):
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = target_transform


class TransferSetImages(Dataset):
    def __init__(self, samples, transform=None, target_transform=None):
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform
        if samples[0][0].shape[0]!=samples[0][0].shape[1]:
            # the image is stored as C x H x W, we need to transpose it as H x W x C before using
            self.data = [self.samples[i][0].transpose([1,2,0]) for i in range(len(self.samples))]
        else:
            self.data = [self.samples[i][0] for i in range(len(self.samples))]
        self.targets = [self.samples[i][1] for i in range(len(self.samples))]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        is_float = isinstance(img, np.ndarray) and img.dtype in (np.float32, np.float64)
        if is_float:
            import torch
            img_t = torch.tensor(img)
            # data was transposed to HWC in __init__, so we transpose back to CHW
            if img_t.ndim == 3 and (img_t.shape[-1] == 3 or img_t.shape[-1] == 1):
                img_t = img_t.permute(2, 0, 1)
            img = img_t
        else:
            img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

class TensorTransferSet(Dataset):
    def __init__(self, samples, transform=None, target_transform=None):
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x, target = self.samples[index]
        # Try to convert tensor back to PIL Image if a standard transform is expected
        if self.transform is not None:
            # In JBTR, the image is already normalized and a float tensor (C, H, W).
            # We might need to map it back to PIL if the `transform` expects PIL.
            # Alternatively, if `transform` handles tensors, just apply it.
            try:
                import torchvision.transforms as T
                # We just return the tensor since JBTR usually returns already pre-processed things. 
                # Or if we strictly need PIL:
                if getattr(self.transform, 'transforms', None) is not None:
                    img = TF.to_pil_image(x)
                    x = self.transform(img)
                else:
                    x = self.transform(x)
            except Exception:
                pass
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        return x, target

    def __len__(self):
        return len(self.samples)

def samples_to_transferset(samples, budget=None, transform=None, target_transform=None):
    # Images are either stored as paths, or numpy arrays
    import torch
    from torch.utils.data import Dataset
    
    assert budget <= len(samples), 'Required {} samples > Found {} samples'.format(budget, len(samples))
    
    if isinstance(samples, Dataset):
        budget_samples = [samples[i] for i in range(budget)]
    else:
        budget_samples = samples[:budget]

    sample_x = budget_samples[0][0]

    if isinstance(sample_x, str):
        return TransferSetImagePaths(budget_samples, transform=transform, target_transform=target_transform)
    elif isinstance(sample_x, np.ndarray):
        return TransferSetImages(budget_samples, transform=transform, target_transform=target_transform)
    elif isinstance(sample_x, torch.Tensor):
        return TensorTransferSet(budget_samples, transform=transform, target_transform=target_transform)
    else:
        raise ValueError('type(x_i) ({}) not recognized. Supported types = (str, np.ndarray, torch.Tensor)'.format(type(sample_x)))


BBOX_CHOICES = ['none', 'topk', 'rounding',
                'reverse_sigmoid', 'reverse_sigmoid_wb',
                'rand_noise', 'rand_noise_wb',
                'mad', 'mad_wb','mld', 'am']
