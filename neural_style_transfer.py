# neural_style_transfer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load the content and style images
def load_image(image_path, max_size=400):
    image = Image.open(image_path).convert('RGB')
    size = max(max(image.size), max_size)
    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    image = transform(image)[:3, :, :].unsqueeze(0)
    return image

# Define the model
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = models.vgg19(pretrained=True).features[:21]
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.features(x)

# Main function for style transfer
def style_transfer(content_img, style_img, num_steps=500, style_weight=1000000, content_weight=1):
    vgg = VGG().to(device).eval()
    content_features = vgg(content_img).detach()
    style_features = vgg(style_img).detach()

    target = content_img.clone().requires_grad_(True).to(device)
    optimizer = optim.Adam([target], lr=0.003)

    for step in range(num_steps):
        target_features = vgg(target)
        content_loss = content_weight * torch.mean((target_features - content_features) ** 2)
        style_loss = style_weight * torch.mean((target_features - style_features) ** 2)

        loss = content_loss + style_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f'Step {step}, Loss: {loss.item()}')

    return target

# Helper function to convert tensor to image
def tensor_to_image(tensor):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = transforms.ToPILImage()(image)
    return image

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

content_img = load_image('path_to_content_image.jpg').to(device)
style_img = load_image('path_to_style_image.jpg').to(device)

output = style_transfer(content_img, style_img)
output_image = tensor_to_image(output)

plt.imshow(output_image)
plt.show()
