# Pix2Pix Implementation in PyTorch

This project is an implementation of the Pix2Pix model using PyTorch. Pix2Pix is a deep learning model designed for image-to-image translation tasks, utilizing a conditional Generative Adversarial Network (cGAN). https://arxiv.org/abs/1611.07004 
I ran it on Facade paired image dataset and the network learns as expected. However, one should search for better hyperparameters for better learning depending on the dataset.

## Features

- Full Pix2Pix architecture implemented from scratch.
- Customizable network parameters for different image translation tasks.
- Supports training with paired image datasets.

## Training
After training it with the default parameters given in this codebase I got some good results with the facade dataset. However, I still think the learning process has been poorly optimized and there should definetely be room for more improvement. One can use this codebase to have fun and take is as the base working architecture for pix2pix tasks. 

## Some Results After Few Hundreds of Epochs Training
<img width="518" height="260" alt="epoch_996_fake" src="https://github.com/user-attachments/assets/a62424f4-5768-4ba5-91d9-a2ebfb1b143c" />
<img width="518" height="260" alt="epoch_952_fake" src="https://github.com/user-attachments/assets/b301f032-d4a1-4a14-83a0-15394190f439" />
<img width="518" height="260" alt="epoch_938_fake" src="https://github.com/user-attachments/assets/4533c393-312a-40db-8136-02765a38cfad" />
<img width="518" height="260" alt="epoch_942_fake" src="https://github.com/user-attachments/assets/8041051d-119e-4ad0-8d47-4dad62220d02" />
<img width="518" height="260" alt="epoch_974_fake" src="https://github.com/user-attachments/assets/816a3395-4a98-41de-93fa-419ffb2cf56e" />
<img width="518" height="260" alt="epoch_974_fake" src="https://github.com/user-attachments/assets/816a3395-4a98-41de-93fa-419ffb2cf56e" />
