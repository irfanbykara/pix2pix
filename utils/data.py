from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class PairedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Expects folder to have subfolders: 'input' and 'target'
        """
        self.input_dir = os.path.join(root_dir, "input")
        self.target_dir = os.path.join(root_dir, "target")
        self.filenames = os.listdir(self.input_dir)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.filenames[idx])
        target_path = os.path.join(self.target_dir, self.filenames[idx])
        input_img = Image.open(input_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        return input_img, target_img

def get_dataloader(root_dir, image_size=256, batch_size=16, num_workers=4, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    dataset = PairedImageDataset(root_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def get_test_dataloader(test_dir, image_size=256, batch_size=1, num_workers=2):
    return get_dataloader(test_dir, image_size=image_size, batch_size=batch_size, num_workers=num_workers, shuffle=False)
