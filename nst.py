# Import modules

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import VGG19_Weights
import matplotlib.pyplot as plt
from torchvision.utils import save_image

li = [0, 5, 10, 19, 28]   # List of layers


device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
print("Task on Device ", device)


class VGG19(nn.Module):

    def __init__(self):
        super(VGG19, self).__init__()  # Call constructor of base class
        self.model = models.vgg19(weights=VGG19_Weights.DEFAULT).features

    def forward(self, x):
        x = x.to(device)
        features = []       # features of required layers
        for index, layer in enumerate(self.model):
            x = layer(x)
            if index in li:
                features.append(x)
        return features


# Initialize the model
model = VGG19()
model = model.to(device)
model.eval()
# print(model.model)

# Load the content, style images as tensors


def load_images(loader, img_name):

    # loader transforms the image
    # img_name is the name of the image
    img = Image.open(img_name)
    img = loader(img)
    img = torch.unsqueeze(img, 0)
    return img

# Parameters


size = 256
crop = 224
n_steps = 2000
learning_rate = 0.03
alpha = 1
beta = 0.01

img_loader = transforms.Compose(
    [transforms.Resize(size),
     transforms.CenterCrop(crop),
     transforms.ToTensor()]
    )

cimg_name = "golden_la.jpeg"
content_img = load_images(img_loader, cimg_name)


# print(content_img.shape)
# Display content image
# plt.imshow((content_img.squeeze(0)).permute(1, 2, 0))
# plt.show()

style_img_name = "stary_night.jpeg"
style_img = load_images(img_loader, style_img_name)


# print(style_img.shape)
# Display style image
# plt.imshow((style_img.squeeze(0)).permute(1, 2, 0))
# plt.show()

# Initialize generated image

gen_image = Image.open(cimg_name)
gen_image.save("gen_img.png")
gen_image_name = "gen_img.png"
gen_img = load_images(img_loader, gen_image_name)
gen_img.requires_grad = True


# print(style_img.shape)
# Display style image
# plt.imshow((gen_img.squeeze(0)).permute(1, 2, 0))
# plt.show()

# We have the content, style and generated image(initialized)
# Now start the training process (Calculate the loss)

# Set optimizer
optimizer = optim.Adam([gen_img], lr=learning_rate)
min_loss = 10e8

for epoch in range(n_steps):
    content_features = model(content_img)
    style_features = model(style_img)
    gen_features = model(gen_img)

    optimizer.zero_grad()
    content_loss = 0
    style_loss = 0

    for layer_content, layer_style, layer_gen in zip(content_features, style_features, gen_features):

        C = layer_content.reshape(1, -1)
        A = layer_gen.reshape(1, -1)
        content_loss += torch.mean((C-A)**2, 1)     # Just Frobenius Norm for Content loss

        S = layer_style.reshape(layer_style.shape[1], -1)
        G = layer_gen.reshape(layer_gen.shape[1], -1)

        # Generate gram matrix
        gram_S = torch.matmul(S, S.t())
        gram_gen = torch.matmul(G, G.t())

        n_channels = layer_content.shape[1]
        height = layer_content.shape[2]
        width = layer_content.shape[3]


        style_loss += (torch.mean((gram_S-gram_gen)**2))

    loss = alpha * content_loss + beta * style_loss
    if loss < min_loss:
        min_loss = loss
        save_image(gen_img[0], "best.png")

    print(f"Loss at epoch {epoch+1} is ", loss)
    loss.backward()
    optimizer.step()

    # print(gen_img[0, 0])


    # Save img the generated image
    if epoch % 200 == 0:
        save_image(gen_img[0], "gen_img"+str(epoch+1)+".png")



























