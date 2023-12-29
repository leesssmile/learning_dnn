import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import os
import time

def gen_outputs(g_model, device, epoch):
    g_model.to(device)
    g_model.eval()

    with torch.no_grad():
        z = torch.randn(16, 100).to(device)
        generated = g_model(z)

    plt.figure(figsize=(4, 4))

    for i in range(0, 16):
        plt.subplot(4, 4, i+1)
        plt.imshow(generated[i].view(28, 28).cpu().numpy(), cmap='gray')
        plt.axis("off")

    plt.savefig("./generated_images/g_model_out_epoch_{:}.png".format(epoch))
    plt.close()

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(100,  256)
        self.fc2 = nn.Linear(256,  512)
        self.fc3 = nn.Linear(512,  1024)
        self.fc4 = nn.Linear(1024, 28*28*1)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)

        self.act  = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.act(x)

        x = self.fc4(x)
        y = self.tanh(x)

        return y

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(28*28*1, 1024)
        self.fc2 = nn.Linear(1024,    512)
        self.fc3 = nn.Linear(512,     256)
        self.fc4 = nn.Linear(256,     1)

        self.act = nn.LeakyReLU(0.2)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten (batch_size, 28*28*1)

        x = self.fc1(x)
        x = self.act(x)

        x = self.fc2(x)
        x = self.act(x)

        x = self.fc3(x)
        x = self.act(x)

        x = self.fc4(x)
        y = self.sig(x)

        return y


device     = torch.device("cuda" if (torch.cuda.is_available() == True) else "cpu")
num_epochs = 200
batch_size = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
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

if (os.path.exists("./g_model/g_model.ckpt")):
    check_point = torch.load("./g_model/g_model.ckpt")
    g_model.load_state_dict(check_point)
    print ("load g_model weights")

if (os.path.exists("./d_model/d_model.ckpt")):
    check_point = torch.load("./d_model/d_model.ckpt")
    d_model.load_state_dict(check_point)
    print ("load d_model weights")

g_model.to(device)
d_model.to(device)

criterion = nn.BCELoss() #Binary Cross-Entropy Loss

g_opt = optim.Adam(g_model.parameters(), betas=(0.5, 0.999), lr=2e-4)
d_opt = optim.Adam(d_model.parameters(), betas=(0.5, 0.999), lr=2e-4)

for epoch in range(1, num_epochs+1):
    for batch_idx, (real_data, _) in enumerate(dataloader):
        start_time = time.time()
        batch_size = len(real_data)

        real_data = real_data.to(device)

        # Train discriminator
        d_real = d_model(real_data) # D(Y)

        z = torch.randn(batch_size, 100).to(device)
        fake_img = g_model(z)        # G(x)
        outputs  = d_model(fake_img) # D(G(x))

        real_label = torch.ones(outputs.shape).to(device)
        fake_label = torch.zeros(outputs.shape).to(device)

        d_loss_real = criterion(d_real, real_label)
        d_loss_fake = criterion(outputs, fake_label)
        d_loss      = (d_loss_real + d_loss_fake) * 0.5

        d_model.zero_grad()
        d_loss.backward(retain_graph=True)
        d_opt.step()

        # Train generator
        z = torch.randn(batch_size, 100).to(device)
        fake_img   = g_model(z)        # G(x)
        outputs    = d_model(fake_img) # D(G(x))

        real_label = torch.ones(outputs.shape).to(device)

        g_loss = criterion(outputs, real_label)

        g_model.zero_grad()
        g_loss.backward()
        g_opt.step()

        if ((batch_idx + 1) % 200 == 0):
            print ("epoch {:}/{:}, batch {:}/{:}, g_loss : {:.3f}, d_loss : {:.3f} (run_time : {:.3f} sec)".format(epoch, num_epochs, batch_idx+1, len(dataloader), g_loss, d_loss, time.time() - start_time))

    if (epoch % 10 == 0):
        gen_outputs(g_model=g_model, device=device, epoch=epoch)

# Save model
if (os.path.exists("./g_model")):
    torch.save(g_model.state_dict(), "./g_model/g_model_epoch_{:}.ckpt".format(num_epochs))
    print ("save g_model weights")

if (os.path.exists("./d_model")):
    torch.save(d_model.state_dict(), "./d_model/d_model_epoch_{:}.ckpt".format(num_epochs))
    print ("save d_model weights")

# Make generator outputs
gen_outputs(g_model=g_model,device=device, epoch=num_epochs)
