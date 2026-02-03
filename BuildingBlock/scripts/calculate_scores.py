import os
import random
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from scipy.linalg import sqrtm

from cleanfid import fid


def calculate_fid(filepaths1, filepaths2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval()
    model = model.to(device)

    preprocess = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def preprocess_images(filepaths):
        images = []
        for filepath in filepaths:
            img = Image.open(filepath).convert("RGB")
            img = preprocess(img)
            img = img.unsqueeze(0)
            images.append(img)
        images = torch.cat(images)
        return images

    images1 = preprocess_images(filepaths1).to(device)
    images2 = preprocess_images(filepaths2).to(device)

    with torch.no_grad():
        features1 = model(images1).cpu().numpy()
        features2 = model(images2).cpu().numpy()

    mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    return fid


file_path_real = "../sample_real"
file_path_fake = "../sample/250115_atiss/merge"

# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# torch.cuda.manual_seed_all(0)
# np.random.seed(0)
# random.seed(0)
fid_score = fid.compute_fid(file_path_real, file_path_fake, device=torch.device("cuda"))
print("fid score:", fid_score)
kid_score = fid.compute_kid(file_path_real, file_path_fake, device=torch.device("cuda"))
print("kid score:", kid_score)
