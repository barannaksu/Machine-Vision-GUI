import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from style_transfer.models.transformer_net import TransformerNet
from style_transfer.utils import gram_matrix

# -------- CONFIG --------
image_size = 256
batch_size = 4
epochs = 2
content_weight = 1e5
style_weight = 1e10
learning_rate = 1e-3

style_image_path = "style_transfer/data/vangogh.jpg"
dataset_path = "style_transfer/data/train_images"
save_model_path = "style_transfer/models/vangogh.pth"

# -------- CUSTOM DATASET --------
class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.image_paths = [os.path.join(folder_path, f)
                            for f in os.listdir(folder_path)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# -------- VGG FEATURE EXTRACTOR --------
class VGGFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential(*[vgg[i] for i in range(4)])     # relu1_2
        self.slice2 = torch.nn.Sequential(*[vgg[i] for i in range(4, 9)])  # relu2_2
        self.slice3 = torch.nn.Sequential(*[vgg[i] for i in range(9, 16)]) # relu3_3
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        return h1, h2, h3

# -------- TRANSFORMS --------
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255)
])

# -------- LOAD DATA --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = ImageFolderDataset(dataset_path, transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -------- STYLE IMAGE --------
style_image = Image.open(style_image_path).convert("RGB")
style_image = transform(style_image).unsqueeze(0).to(device)

vgg = VGGFeatures().to(device)
style_features = vgg(style_image)
style_grams = [gram_matrix(f) for f in style_features]

# -------- TRAINING SETUP --------
model = TransformerNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
mse_loss = torch.nn.MSELoss()

print("🧠 Training started...")

for epoch in range(epochs):
    model.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in pbar:
        batch = batch.to(device)

        optimizer.zero_grad()
        output = model(batch)
        output = torch.clamp(output, 0, 255)

        # VGG features
        output_features = vgg(output)
        content_features = vgg(batch)

        # Content loss (relu2_2)
        content_loss = mse_loss(output_features[1], content_features[1])

        # Style loss
        style_loss = 0
        for o, s in zip(output_features, style_grams):
            gram_o = gram_matrix(o)
            style_loss += mse_loss(gram_o, s.expand_as(gram_o))

        total_loss = content_weight * content_loss + style_weight * style_loss
        total_loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=total_loss.item())

# -------- SAVE MODEL --------
os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
torch.save(model.state_dict(), save_model_path)
print(f"✅ Model saved to: {save_model_path}")
