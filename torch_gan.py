import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import os
import time

device = torch.device("cuda" if (torch.cuda.is_available() == True) else "cpu")

num_epochs = 500
batch_size = 64

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.Sigmoid(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.Linear(256, 28*28*1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28*1, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

mnist_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)

g_model = Generator()
d_model = Discriminator()

if (os.path.exists("./g_model_torch/g_model.ckpt")):
    check_point = torch.load("./g_model_torch/g_model.ckpt")
    g_model.load_state_dict(check_point)
    print ("load g_model weights")

if (os.path.exists("./d_model_torch/d_model.ckpt")):
    check_point = torch.load("./d_model_torch/d_model.ckpt")
    d_model.load_state_dict(check_point)
    print ("load d_model weights")


g_model.to(device)
d_model.to(device)

criterion = nn.BCELoss() #Binary Cross-Entropy Loss

g_opt = optim.Adam(g_model.parameters(), lr=1e-4)
d_opt = optim.Adam(d_model.parameters(), lr=1e-4)

for epoch in range(1, num_epochs+1):
    for batch_idx, (real_data, _) in enumerate(dataloader):
        start_time = time.time()

        #real_data = real_data.reshape(batch_size, -1).to(device)
        real_data = real_data.to(device)

        # To prevent size overflow or underflow
        batch_size = len(real_data)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train the d_model
        d_opt.zero_grad()
        d_real = d_model(real_data)
        real_loss = criterion(d_real, real_labels)

        z = torch.randn(batch_size, 100).to(device)
        fake_img = g_model(z)
        outputs  = d_model(fake_img)
        fake_loss = criterion(outputs, fake_labels)

        # real_loss  = -1 * torch.log(d_model(real_data))
        # fake_loss  = -1 * torch.log(1.0 - d_model(fake_img))
        # d_loss     = (real_loss + fake_loss).mean()

        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_opt.step()

        # Train the g_model
        g_opt.zero_grad()

        z = torch.randn(batch_size, 100).to(device)
        fake_img = g_model(z)
        outputs  = d_model(fake_img)
        g_loss   = criterion(outputs, real_labels)

        # g_loss = -1 * torch.log(1.0 - d_model(fake_img)).mean()

        g_loss.backward()
        g_opt.step()

        if ((batch_idx + 1) % 200 == 0):
            print ("epoch {:}/{:}, batch {:}/{:}, g_loss : {:.3f}, d_loss : {:.3f} (run_time : {:.3f} sec)".format(epoch, num_epochs, batch_idx+1, len(dataloader), g_loss.item(), d_loss.item(), time.time() - start_time))

# Save model
if (os.path.exists("./g_model_torch")):
    torch.save(g_model.state_dict(), "./g_model_torch/g_model.ckpt")
    print ("save g_model")

if (os.path.exists("./d_model_torch")):
    torch.save(d_model.state_dict(), "./d_model_torch/d_model.ckpt")
    print ("save d_model")

# Make generator outputs
g_model.eval()
with torch.no_grad():
    z = torch.randn(16, 100).to(device)
    generated_images = g_model(z)

plt.figure(figsize=(4, 4))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(generated_images[i].view(28, 28).cpu().numpy(), cmap='gray')
    plt.axis("off")

plt.show()
#plt.savefig("./generated_images/torch_g_imgs_{:0d}_epochs.png".format(num_epochs))
#plt.close("all")
