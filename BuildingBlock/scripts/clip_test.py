import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = (
    preprocess(
        Image.open("../3D-FUTURE-model/410561db-f2a3-4092-992e-9b1e8154d01b/image.jpg")
    )
    .unsqueeze(0)
    .to(device)
)
text = clip.tokenize(
    ["Grey bed with red quilt", "Red bed with gray quilt", "Grey bed with blue quilt"]
).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    # print(image_features)
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
