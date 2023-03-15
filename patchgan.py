# Code Adapted from https://pennylane.ai/qml/demos/tutorial_quantum_gans.html
# Library imports
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pennylane as qml

# Pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from scipy.linalg import sqrtm
from torchvision import datasets, transforms

from PIL import Image
from matplotlib import cm

# Set the random seed for reproducibility
from torchvision.transforms import InterpolationMode, Resize

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', dest='mode', type=str, default='train')
parser.add_argument('--tr1', dest='trained_model1', type=str, default='None')
parser.add_argument('--tr2', dest='trained_model2', type=str, default='None')
parser.add_argument('--ds', dest='dataset', type=str, default='MNIST')
parser.add_argument('--ds_class', dest='ds_class', type=int, default=5)
parser.add_argument('--env', dest='env', type=str, default='simulation')


args = parser.parse_args()
def calculate_fid(act1, act2):
    act1 = act1.numpy()
    act2 = act2.numpy()
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


######################################################################
# Data
# ~~~~
#


######################################################################
# As mentioned in the introduction, we will use a `small
# dataset <https://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits>`__
# of handwritten zeros. First, we need to create a custom dataloader for
# this dataset.
#


class DigitsDataset(Dataset):
    """Pytorch dataloader for the Optical Recognition of Handwritten Digits Data Set"""

    def __init__(self, csv_file, label=0, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = csv_file
        self.transform = transform
        self.df = self.filter_by_label(label)

    def filter_by_label(self, label):
        # Use pandas to return a dataframe of only zeros
        df = pd.read_csv(self.csv_file)
        df = df.loc[df.iloc[:, -1] == label]
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.df.iloc[idx, :-1] / 16
        image = np.array(image)
        image = image.astype(np.float32).reshape(8, 8)

        if self.transform:
            image = self.transform(image)

        # Return image and label
        return image, 0


######################################################################
# Next we define some variables and create the dataloader instance.
#

image_size = 8  # Height / width of the square images
batch_size = 8

transform = transforms.Compose([transforms.ToTensor()])
dataset = DigitsDataset(csv_file="optdigits.tra", transform=transform)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, drop_last=True
)

if args.dataset == 'MNIST':
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist',
                                                          download=True,
                                                          train=True,

                                                          transform=transforms.Compose([
                                                              torchvision.transforms.ToTensor(),
                                                              transforms.Lambda(Resize(8)), # Removing this line makes the images full size
                                                          ])),
                                           batch_size=10000,
                                           shuffle=True)
elif args.dataset == 'Fashion':
    train_loader = torch.utils.data.DataLoader(datasets.FashionMNIST('../fashion',
                                                          download=True,
                                                          train=True,

                                                          transform=transforms.Compose([
                                                              torchvision.transforms.ToTensor(),
                                                              transforms.Lambda(Resize(8)), # Removing this line makes the images full size
                                                          ])),
                                           batch_size=10000,
                                           shuffle=True)

train_data = []
label_to_keep = args.ds_class
for (data, labels) in train_loader:
    for x, y in zip(data, labels):
        if y == label_to_keep:
            train_data.append(x.numpy())
train_data = torch.tensor(train_data)
last_sample_to_keep = train_data.shape[0] - (train_data.shape[0] % batch_size)
train_data = train_data[:last_sample_to_keep]
train_data = train_data.reshape([-1, batch_size, image_size, image_size])
print(train_data[0])
        ######################################################################


# Implementing the Discriminator
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# For the discriminator, we use a fully connected neural network with two
# hidden layers. A single output is sufficient to represent the
# probability of an input being classified as real.
#


class Discriminator(nn.Module):
    """Fully connected classical discriminator"""

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # Inputs to first hidden layer (num_input_features -> 64)
            nn.Linear(image_size * image_size, 64),
            nn.ReLU(),
            # First hidden layer (64 -> 16)
            nn.Linear(64, 16),
            nn.ReLU(),
            # Second hidden layer (16 -> output)
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


# Quantum variables
n_qubits = 5  # Total number of qubits / N
n_a_qubits = 1  # Number of ancillary qubits / N_A
q_depth = 6  # Depth of the parameterised quantum circuit / D
n_generators = 4  # Number of subgenerators for the patch method / N_G

######################################################################
# Now we define the quantum device we want to use, along with any
# available CUDA GPUs (if available).
#

# Quantum simulator
dev = qml.device("lightning.qubit", wires=n_qubits)
# Enable CUDA device if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


######################################################################
# Next, we define the quantum circuit and measurement process described above.
@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def quantum_circuit(noise, weights):
    weights = weights.reshape(q_depth, n_qubits)

    # Initialise latent vectors
    for i in range(n_qubits):
        qml.RY(noise[i], wires=i)

    # Repeated layer
    for i in range(q_depth):
        # Parameterised layer
        for y in range(n_qubits):
            qml.RY(weights[i][y], wires=y)

        # Control Z gates
        for y in range(n_qubits - 1):
            qml.CZ(wires=[y, y + 1])

    return qml.probs(wires=list(range(n_qubits)))


# For further info on how the non-linear transform is implemented in Pennylane
# https://discuss.pennylane.ai/t/ancillary-subsystem-measurement-then-trace-out/1532
def partial_measure(noise, weights):
    # Non-linear Transform
    probs = quantum_circuit(noise, weights)
    probsgiven0 = probs[: (2 ** (n_qubits - n_a_qubits))]
    probsgiven0 /= torch.sum(probs)

    # Post-Processing
    probsgiven = probsgiven0 / torch.max(probsgiven0)
    return probsgiven


######################################################################
# Now we create a quantum generator class to use during training.

class PatchQuantumGenerator(nn.Module):
    """Quantum generator class for the patch method"""

    def __init__(self, n_generators, q_delta=1):
        """
        Args:
            n_generators (int): Number of sub-generators to be used in the patch method.
            q_delta (float, optional): Spread of the random distribution for parameter initialisation.
        """

        super().__init__()

        self.q_params = nn.ParameterList(
            [
                nn.Parameter(q_delta * torch.rand(q_depth * n_qubits), requires_grad=True)
                for _ in range(n_generators)
            ]
        )
        self.n_generators = n_generators

    def forward(self, x):
        # Size of each sub-generator output
        patch_size = 2 ** (n_qubits - n_a_qubits)
        # Create a Tensor to 'catch' a batch of images from the for loop. x.size(0) is the batch size.
        images = torch.Tensor(x.size(0), 0).to(device)

        # Iterate over all sub-generators
        for params in self.q_params:

            # Create a Tensor to 'catch' a batch of the patches from a single sub-generator
            patches = torch.Tensor(0, patch_size).to(device)
            # for b in batch basically
            for elem in x:
                q_out = partial_measure(elem, params).float().unsqueeze(0)
                patches = torch.cat((patches, q_out))
            # Each batch of patches is concatenated with each other to create a batch of images
            images = torch.cat((images, patches), 1)
        return images


######################################################################
# Training
# ~~~~~~~~
#

######################################################################

def train():
    # Let's define learning rates and number of iterations for the training process.

    lrG = 0.3  # Learning rate for the generator
    lrD = 0.01  # Learning rate for the discriminator
    num_iter = 1000  # Number of training iterations

    ######################################################################
    # Now putting everything together and executing the training process.

    discriminator = Discriminator().to(device)
    generator = PatchQuantumGenerator(n_generators).to(device)

    # Binary cross entropy
    criterion = nn.BCELoss()

    # Optimisers
    optD = optim.SGD(discriminator.parameters(), lr=lrD)
    optG = optim.SGD(generator.parameters(), lr=lrG)

    real_labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
    fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)

    # Fixed noise allows us to visually track the generated images throughout training
    fixed_noise = torch.rand(8, n_qubits, device=device) * math.pi / 2
    num_img_to_save = 100
    save_image_noise = torch.rand(num_img_to_save, n_qubits, device=device) * math.pi / 2

    # Iteration counter
    counter = 0

    # Collect images for plotting later
    results = []
    real_images = []
    generated_images = []
    while True:
        for i, (data) in enumerate(train_data):
        #for i, (data, _) in enumerate(dataloader):


            # Data for training the discriminator
            data = data.reshape(-1, image_size * image_size)
            real_data = data.to(device)
            real_images.append(data)

            # Noise following a uniform distribution in range [0,pi/2)
            noise = torch.rand(batch_size, n_qubits, device=device) * math.pi / 2
            fake_data = generator(noise)

            # Training the discriminator
            discriminator.zero_grad()
            outD_real = discriminator(real_data).view(-1)
            outD_fake = discriminator(fake_data.detach()).view(-1)

            errD_real = criterion(outD_real, real_labels)
            errD_fake = criterion(outD_fake, fake_labels)
            # Propagate gradients
            errD_real.backward()
            errD_fake.backward()

            errD = errD_real + errD_fake
            optD.step()

            # Training the generator
            generator.zero_grad()
            outD_fake = discriminator(fake_data).view(-1)
            errG = criterion(outD_fake, real_labels)
            errG.backward()
            optG.step()

            counter += 1
            print(f"Counter: {counter}")
            # Show loss values
            if counter % 10 == 0:

                # print(f'Iteration: {counter}, Discriminator Loss: {errD:0.3f}, Generator Loss: ' + str(errG))
                test_images = generator(fixed_noise).view(8, 1, image_size, image_size).cpu().detach()
                real_images = []
                print(errD, errG)
                # Save images every 50 iterations
                if counter % 50 == 0:
                    results.append(test_images)

                if counter != 0:

                    im = np.reshape(test_images[0], [8, 8])

                    '''
                    Save the batch of 8x8 Image Matrices here with something like np.save('name_of_file', im)
                    '''
                    if counter == 500 or counter == 1000:
                        # Generate 100 images and save them
                        images_to_save = generator(save_image_noise).view(num_img_to_save, image_size, image_size).cpu().detach()
                        save_images(images_to_save, str(counter))

                    im = Image.fromarray(np.uint8(cm.gist_earth(im) * 255))
                    im = im.resize((28, 28), resample=Image.BOX)

                    im.save(f"patchgan.png")
                if counter >= num_iter:
                    break

        print("Made it out of the loop")
        if counter >= num_iter:
            break

    fig = plt.figure(figsize=(10, 5))
    outer = gridspec.GridSpec(5, 2, wspace=0.1)

    for i, images in enumerate(results):
        inner = gridspec.GridSpecFromSubplotSpec(1, images.size(0),
                                                 subplot_spec=outer[i])

        images = torch.squeeze(images, dim=1)
        for j, im in enumerate(images):

            ax = plt.Subplot(fig, inner[j])
            ax.imshow(im.numpy(), cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                # ax.set_title(f'Iteration {50+i*50}', loc='left')
                ax.set_title(str(50 + i * 50), loc='left')
            fig.add_subplot(ax)

    plt.show()


def save_images(images, counter_str):
    def img_save_str(counter_str, method_str, index_str):
        return f"resized_images/count-{counter_str}-mode-{method_str}-index-{index_str}"

    method = InterpolationMode.BILINEAR
    # Had to unsqueeze and squeeze for resizing to work
    for i in range(len(images)):
        im_to_save = images[i]
        resized_img_arr = transforms.Resize(size=28, interpolation=method)(torch.tensor(im_to_save).unsqueeze(0))
        np.save(img_save_str(counter_str, method, i), resized_img_arr)
        resized_image = Image.fromarray(np.uint8(cm.gist_earth(resized_img_arr.squeeze(0)) * 255))
        resized_image.save(img_save_str(counter_str, method, i) + ".png")
        np.save(img_save_str(counter_str, "original", i), im_to_save)
        original_image = Image.fromarray(np.uint8(cm.gist_earth(im_to_save) * 255))
        original_image.save(img_save_str(counter_str, "original",i ) + ".png")


train()
