# Library imports
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pennylane as qml
from torchvision import datasets, transforms
# Pytorch imports
import torch
from sklearn.metrics import explained_variance_score
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from matplotlib import cm
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from itertools import chain
# Set the random seed for reproducibility
from sklearn.preprocessing import StandardScaler
def scale_data(data, scale=None, dtype=np.float32):
    """
    Scales every element in data linearly such that its minimum value is at the bottom of the specified scale,
    and the maximum value is at the top.
    :param data: Numpy array of data to scale.
    :param scale: A 2-element array, whose first value gives lower scale, second gives upper.
    :param dtype: Data type to transform scaled data to.
    :return: Data scaled as specified in the given type.
    """
    if scale is None:
        scale = [-1, 1]
    min_data, max_data = [float(np.min(data)), float(np.max(data))]
    min_scale, max_scale = [float(scale[0]), float(scale[1])]
    data = ((max_scale - min_scale) * (data - min_data) / (max_data - min_data)) + min_scale
    #print(min_data, max_data)
    return data.astype(dtype)

train_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist', 
                                                          download=True, 
                                                          train=True,
                                                          
                                                          transform=transforms.Compose([
                                                              torchvision.transforms.ToTensor(),
                                                              transforms.Lambda(torch.flatten),
                                                          ])), 
                                           batch_size=10000, 
                                           shuffle=True)
# train_loader = torch.utils.data.DataLoader(datasets.FashionMNIST('../fashion', 
#                                                           download=True, 
#                                                           train=True,
                                                          
#                                                           transform=transforms.Compose([
#                                                               torchvision.transforms.ToTensor(),
#                                                               transforms.Lambda(torch.flatten),
#                                                           ])), 
#                                            batch_size=10000, 
#                                            shuffle=True)
train_data = []
label_to_keep = 8
label_to_keep_name = str(label_to_keep) + "_" #+ 'F'
for (data, labels) in train_loader:
  for x, y in zip(data, labels):
    if y == label_to_keep:
      train_data.append(x.numpy())
# from PIL import Image 
# for i in range(30):
#     im = np.reshape(train_data[i], [28, 28])
#     im = Image.fromarray(np.uint8(cm.gist_earth(im)*255))
#     im = im.save(f"gen_images\\real_{label_to_keep}_{i}.png")
# exit()
def calculate_fid(act1, act2):

    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
train_data = scale_data(np.array(train_data), [0,1])
image_size = 5  # Height / width of the square images
batch_size = 8
pca_dims=40
n_qubits = 5  # Total number of qubits / N
n_a_qubits = 1  # Number of ancillary qubits / N_A
q_depth = 6  # Depth of the parameterised quantum circuit / D
n_generators = 8  # Number of subgenerators for the patch method / N_G
pca = PCA(n_components=pca_dims)
pca_data_full = pca.fit_transform(train_data)
print(pca.explained_variance_ratio_)
ordering = []
for i in range(8):
    k = 4* i
    l = [i, 39-k, 38-k, 37-k, 36-k]
    ordering.append(l)

#exit()
#print(sum(pca.explained_variance_ratio_))
pca_min, pca_max = np.min(pca_data_full), np.max(pca_data_full)
inversed = pca.inverse_transform(pca_data_full)
#plt.imshow(np.reshape(inversed[5], [28, 28]))
#plt.imshow(np.reshape(train_data[5], [28, 28]))
#plt.show()
print(np.min(inversed), np.max(inversed))
#rand_stuff = pca.inverse_transform(np.random.randn(8, 16))
#x = calculate_fid(rand_stuff, inversed)
#print(x)
#exit() 
full_train_data = [(i,j) for i,j in zip(scale_data(pca_data_full), train_data)]
#full_train_data = [(i,j) for i,j in zip(pca_data, train_data)]

transform = transforms.Compose([transforms.ToTensor()])
dataloader = torch.utils.data.DataLoader(
    scale_data(pca_data_full), batch_size=batch_size, shuffle=True, drop_last=True
)

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
            nn.Linear(pca_dims, 64),
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

class PatchDiscriminator(nn.Module):
    """Fully connected classical discriminator"""

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # Inputs to first hidden layer (num_input_features -> 64)
            nn.Linear(pca_dims, 64),
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

    #weights = weights.reshape(q_depth, n_qubits)

    # Initialise latent vectors
    for i in range(n_qubits):
        qml.RY(noise[i], wires=i)
        qml.RX(noise[i], wires=i)

    qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))
    # # Repeated layer
    # for i in range(q_depth):
    #     # Parameterised layer
    #     for y in range(n_qubits):
    #         qml.RY(weights[i][y], wires=y)

    #     # Control Z gates
    #     for y in range(n_qubits - 1):
    #         qml.CZ(wires=[y, y + 1])

    return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
    #return qml.probs(wires=list(range(n_qubits)))

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def quantum_circuit_old(noise, weights):

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

    return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]

# For further info on how the non-linear transform is implemented in Pennylane
# https://discuss.pennylane.ai/t/ancillary-subsystem-measurement-then-trace-out/1532
def partial_measure(noise, weights):
    # Non-linear Transform
    probs = quantum_circuit_old(noise, weights)
    #probsgiven0 = probs[: (2 ** (n_qubits - n_a_qubits))]
    #probsgiven0 = probs[:n_qubits - n_a_qubits]
    #probsgiven0 /= torch.sum(probs)

    # Post-Processing
    #probsgiven = probsgiven0 / torch.max(probsgiven0)
    return probs
    return probsgiven0

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
                nn.Parameter(q_delta * torch.rand(q_depth, n_qubits), requires_grad=True)
                for _ in range(n_generators)
            ]
        )

        self.n_generators = n_generators

    def forward(self, x):
        images = []
        patch_size = image_size
        images = torch.Tensor(x.size(0), 0).to(device)

        for params in self.q_params:
            # Create a Tensor to 'catch' a batch of the patches from a single sub-generator
            patches = torch.Tensor(0, patch_size).to(device)
            # for b in batch basically
            for elem in x:
                q_out = partial_measure(elem, params).float().unsqueeze(0)
                patches = torch.cat((patches, q_out))
            # Each batch of patches is concatenated with each other to create a batch of images
            #patches = torch.transpose(patches, )
            flattened_order =  [j for sub in ordering for j in sub]
            patches = torch.flatten(patches)
            patches = patches[flattened_order]
            patches = patches.reshape(batch_size, patch_size)
            images = torch.cat((images, patches), 1)
        return images



######################################################################
# Training
# ~~~~~~~~
#

######################################################################
# Let's define learning rates and number of iterations for the training process.

lrG = 0.3  # Learning rate for the generator
lrD = 0.05  # Learning rate for the discriminator
num_iter = 500  # Number of training iterations

gen_losses = []
disc_losses = []
######################################################################
# Now putting everything together and executing the training process.

discriminator = Discriminator().to(device)
generator = PatchQuantumGenerator(n_generators).to(device)
#generator = PatchQuantumGeneratorTranspose(n_generators).to(device)

# Binary cross entropy
criterion = nn.BCELoss()

# Optimisers
optD = optim.SGD(discriminator.parameters(), lr=lrD)
optG = optim.SGD(generator.parameters(), lr=lrG)

real_labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)

# Fixed noise allows us to visually track the generated images throughout training
fixed_noise = torch.rand(8, n_qubits, device=device) * math.pi / 2

# Iteration counter
counter = -1

noise_upper_bound = math.pi/8

def relu(x):
    return x * (x > 0)
def get_noise_upper_bound(gen_loss, disc_loss, original_ratio):
    R = disc_loss.detach().numpy()/gen_loss.detach().numpy()
    beta = .7
    return math.pi/8 + (5 *math.pi / 8) * relu(np.tanh((R - (beta*original_ratio))))
original_ratio = None
upper_bounds = [math.pi/8]
# Collect images for plotting later
results = []
fids = []
generated_images = []
while True:
    for e in range(10):
        for i, train_pair in enumerate(dataloader):
            pca_data = train_pair
            # Data for training the discriminator
            data = pca_data.reshape(batch_size, pca_dims)
            real_data = data.to(device).to(torch.float32)

            # Noise follwing a uniform distribution in range [0,pi/2)
            
            noise = torch.rand(batch_size, n_qubits, device=device) * noise_upper_bound
            fake_data = generator(noise)

            discriminator.zero_grad()

            outD_real = discriminator(real_data).view(-1)
            outD_fake = discriminator(fake_data.detach()).view(-1)


            errD_real = criterion(outD_real, real_labels)
            errD_fake = criterion(outD_fake, fake_labels)
            # Propagate gradients
            errD_real.backward()
            errD_fake.backward()
            errG = criterion(outD_fake, real_labels)


            errD = errD_real + errD_fake
            gen_losses.append(errG.detach().numpy())
            disc_losses.append(errD.detach().numpy())

            if counter == -1:
                test_images = generator(noise).detach().numpy()
                fid2 = calculate_fid(test_images, pca_data_full)

                test_images = pca.inverse_transform(test_images)
                print(np.min(test_images), np.max(test_images), np.mean(train_data), np.mean(test_images))
                fid = calculate_fid(test_images.reshape([batch_size, 784]), train_data)
                fids.append(fid2)
                test_images = scale_data(test_images,[0,1])
                real_images = []
                print(fid, fid2)
            #if i%10 == 0:
            optD.step()

            # Training the generator
            generator.zero_grad()
            outD_fake = discriminator(fake_data).view(-1)
            errG = criterion(outD_fake, real_labels)
            errG.backward()
            optG.step()
            if original_ratio is None:
                original_ratio = errD.detach().numpy()/errG.detach().numpy()
            noise_upper_bound = get_noise_upper_bound(errG, errD, original_ratio)
            upper_bounds.append(noise_upper_bound)
            np.save(f'upper_bounds_{label_to_keep_name}', upper_bounds)
            np.save(f'fids_{label_to_keep_name}', fids)
            #print(f'Iteration: {counter}, Discriminator Loss: {errD:0.3f}, Generator Loss: ' + str(errG))
            counter += 1

            # Show loss values         
            if counter % 10 == 0:
                
                test_images = generator(noise).detach().numpy()
                fid2 = calculate_fid(test_images, pca_data_full)

                test_images = pca.inverse_transform(test_images)
                fid = calculate_fid(test_images.reshape([batch_size, 784]), train_data)
                test_images = scale_data(test_images,[0,1])
                real_images = []
                print(label_to_keep, errD/errG, fid2)

                np.save(f'gen_loss_{label_to_keep_name}', gen_losses)
                np.save(f'disc_loss_{label_to_keep_name}', disc_losses)
                from PIL import Image 
                im = np.reshape(test_images[0], [28, 28])
                im = Image.fromarray(np.uint8(cm.gist_earth(im)*255))
                im = im.save(f"gen_images_dist\\{label_to_keep_name}_{counter}.png")
                # Save images every 50 iterations
                if counter % 50 == 0:
                    results.append(test_images)
            # if counter == 0:
            #     #plt.imshow(test_images[-1].reshape(28, 28, 1))
            #     #plt.show() 
            #     plt.imshow(test_images[-1].reshape(28, 28, 1))
            #     plt.show()
            #     break
        if counter == num_iter:
            break




# fig = plt.figure(figsize=(10, 5))
# outer = gridspec.GridSpec(5, 2, wspace=0.1)

# for i, images in enumerate(results):
#     inner = gridspec.GridSpecFromSubplotSpec(1, images.size(0),
#                     subplot_spec=outer[i])
    
#     images = torch.squeeze(images, dim=1)
#     for j, im in enumerate(images):

#         ax = plt.Subplot(fig, inner[j])
#         ax.imshow(im.numpy(), cmap="gray")
#         ax.set_xticks([])
#         ax.set_yticks([])
#         if j==0:
#             #ax.set_title(f'Iteration {50+i*50}', loc='left')
#             ax.set_title(str(50+i*50), loc='left')
#         fig.add_subplot(ax)



