import torch
from anomalib.models import Padim
from torchvision import transforms
from PIL import Image
import numpy as np


def load_model(weights_path: str) -> Padim:
    model = Padim(
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
        pre_trained=False
    )
    state_dict = torch.load(weights_path, map_location="cpu")
    model.model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(image_path: str, image_size: int = 256) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)  # [1, C, H, W]


def infer(model: Padim, image_tensor: torch.Tensor):
    with torch.no_grad():
        output = model(image_tensor)
        score = output.pred_score.item()
        mask = output.pred_mask

        if mask is None:
            # Eğer maske gelmezse sıfırlarla dolu bir fallback maske oluştur
            mask = torch.zeros((256, 256), dtype=torch.float32)
        else:
            # Torch tensor → NumPy array (bool olabiliyor!)
            mask = mask.squeeze().numpy()

            # Eğer maske boolean ise, normalize etmek için float'a çevir
            if mask.dtype == bool:
                mask = mask.astype(np.float32)

            # Normalize et: [0, 1] aralığına getir (görselleştirme için önemli)
            mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask) + 1e-8)

    return score, mask


# Opsiyonel: Debug amaçlı direkt çalıştırmak için
def show_results(image_path, mask, score):
    import matplotlib.pyplot as plt

    img = Image.open(image_path).convert("RGB").resize((256, 256))
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img)
    axs[0].set_title("Input Image")
    axs[1].imshow(mask, cmap="jet")
    axs[1].set_title(f"Anomaly Map\nScore: {score:.4f}")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = "/Users/baranaksu/Desktop/ytu_yuksek/24_25_bahar/yapay_gorme/HW3/MVTec/bottle/test/contamination/000.png"
    model_path = "padim_bottle.pth"

    model = load_model(model_path)
    image_tensor = preprocess_image(image_path)
    score, mask = infer(model, image_tensor)
    show_results(image_path, mask, score)
