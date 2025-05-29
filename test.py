import torch
from torchvision.utils import save_image
from models.generator import Generator
from utils.facade_dataloader import get_test_dataloader  # You must implement this

import os
def test(model_path, save_dir):
    G = Generator([64, 128, 256, 512, 512, 512, 512], 512)
    G.load_state_dict(torch.load(model_path, map_location='cpu'))
    G.eval()

    dataloader = get_test_dataloader()
    with torch.no_grad():
        for idx, (input_image, _) in enumerate(dataloader):
            fake = G(input_image)
            save_image(fake, f"{save_dir}/output_{idx}.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="facades_label2photo.pth")
    parser.add_argument('--test_dir', type=str, default='dataset/val')
    parser.add_argument('--save_dir', type=str, default='test_outputs')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    test(args.model_path, args.save_dir)
