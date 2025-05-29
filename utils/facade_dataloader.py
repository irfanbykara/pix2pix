from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os

class FacadesTestDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        w, h = image.size
        input_image = image.crop((0, 0, w // 2, h))  # Left side
        return self.transform(input_image), self.image_paths[idx]

    def __len__(self):
        return len(self.image_paths)

def get_test_dataloader():
    dataset = FacadesTestDataset("datasets/facades/test")
    return DataLoader(dataset, batch_size=1)

class FacadesTrainDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1,1]
        ])
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        w, h = image.size
        target_image = image.crop((0, 0, w // 2, h))      # Left side
        input_image = image.crop((w // 2, 0, w, h))      # Right side
       

        return self.transform(input_image), self.transform(target_image)

    def __len__(self):
        return len(self.image_paths)

def get_train_dataloader(batch_size=2, shuffle=True):
    dataset = FacadesTrainDataset("/workspace/irfan_mvp/pix2pix/datasets/facades/train")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
