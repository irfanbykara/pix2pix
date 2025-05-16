import torch
from models.generator import Generator
from models.discriminator import Discriminator
from losses.gan_losses import GeneratorLoss, DiscriminatorLoss
from utils.data import get_dataloader  # Assume you implement this
from torchvision.utils import save_image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(epochs, dataloader, save_dir):
    G = Generator([64, 128, 256, 512, 512, 512, 512], 512).to(device)
    D = Discriminator().to(device)

    g_criterion = GeneratorLoss().to(device)
    d_criterion = DiscriminatorLoss().to(device)

    g_optimizer = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for i, (input_image, real_image) in enumerate(dataloader):
            input_image = input_image.to(device)
            real_image = real_image.to(device)

            fake_image = G(input_image)
            fake_pair = torch.cat([fake_image, input_image], dim=1)
            real_pair = torch.cat([real_image, input_image], dim=1)

            # Update discriminator
            D.zero_grad()
            d_loss = d_criterion(D(fake_pair.detach()), D(real_pair))
            d_loss.backward()
            d_optimizer.step()

            # Update generator
            G.zero_grad()
            g_loss = g_criterion(fake_image, real_image, D(fake_pair))
            g_loss.backward()
            g_optimizer.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}] Step [{i}] G Loss: {g_loss.item():.4f} D Loss: {d_loss.item():.4f}")

        save_image(fake_image, os.path.join(save_dir, f"epoch_{epoch}_fake.png"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset/train')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_dir', type=str, default='outputs')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    dataloader = get_dataloader(args.data_dir, batch_size=args.batch_size)
    train(args.epochs, dataloader, args.save_dir)
