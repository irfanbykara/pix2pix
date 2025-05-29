import torch
from models.generator import Generator
from models.discriminator import Discriminator
from losses.gan_losses import GeneratorLoss, DiscriminatorLoss
from utils.data import get_dataloader  # Assume you implement this
from torchvision.utils import save_image
import os
from utils.facade_dataloader import get_train_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(epochs, dataloader, save_dir):
    generator = Generator([64, 128, 256, 512, 512, 512, 512], 512).to(device)
    discriminator = Discriminator().to(device)

    g_criterion = GeneratorLoss().to(device)
    d_criterion = DiscriminatorLoss().to(device)

    g_optimizer = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=100, gamma=0.5)
    d_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=100, gamma=0.5)

    for epoch in range(epochs):
        for i, (input_image, real_image) in enumerate(dataloader):
            input_image = input_image.to(device)
            real_image = real_image.to(device)

            fake_image = generator(input_image)
            fake_pair = torch.cat([fake_image, input_image], dim=1)
            real_pair = torch.cat([real_image, input_image], dim=1)

            # Update discriminator
            discriminator.zero_grad()
            d_loss = d_criterion(D(fake_pair.detach()), D(real_pair))
            d_loss.backward()
            d_optimizer.step()

            # Update generator
            generator.zero_grad()
            g_loss = g_criterion(fake_image, real_image, D(fake_pair))
            g_loss.backward()
            g_optimizer.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}] Step [{i}] G Loss: {g_loss.item():.4f} D Loss: {d_loss.item():.4f}")
        # Step learning rate schedulers
        g_scheduler.step()
        d_scheduler.step()

        save_image(fake_image, os.path.join(save_dir, f"epoch_{epoch}_fake.png"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset/train')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--save_dir', type=str, default='outputs')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    # dataloader = get_dataloader(args.data_dir, batch_size=args.batch_size)
    dataloader = get_train_dataloader()
    train(args.epochs, dataloader, args.save_dir)
