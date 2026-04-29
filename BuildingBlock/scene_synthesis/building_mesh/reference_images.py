"""Part-level text-to-image reference image generation."""

import base64
import json
import os
import urllib.request
from pathlib import Path

from .prompting import NEGATIVE_PROMPT


FLUX_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
OPENAI_GPT_IMAGE_1_5_MODEL = "gpt-image-1.5"
SILICONFLOW_QWEN_IMAGE_MODEL = "Qwen/Qwen-Image"


class FluxSchnellProvider:
    name = "flux_schnell"

    def __init__(self, model_id=FLUX_MODEL_ID, device="cuda"):
        self.model_id = model_id
        self.device = device
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is None:
            import torch
            from diffusers import FluxPipeline

            pipe = FluxPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
            )
            self._pipeline = pipe.to(self.device)
        return self._pipeline

    def generate(
        self,
        prompt,
        negative_prompt,
        output_path,
        seed=1234,
        width=768,
        height=768,
        num_inference_steps=4,
        guidance_scale=0.0,
    ):
        import torch

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator(self.device).manual_seed(int(seed))
        pipe = self._load_pipeline()
        image = pipe(
            prompt=prompt,
            # FLUX schnell ignores classic negative prompts; keep it in metadata.
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        image.save(output_path)
        return str(output_path)


class SdxlProvider:
    name = "sdxl"

    def __init__(self, model_id=SDXL_MODEL_ID, device="cuda"):
        self.model_id = model_id
        self.device = device
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is None:
            import torch
            from diffusers import StableDiffusionXLPipeline

            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )
            self._pipeline = pipe.to(self.device)
        return self._pipeline

    def generate(
        self,
        prompt,
        negative_prompt,
        output_path,
        seed=1234,
        width=768,
        height=768,
        num_inference_steps=30,
        guidance_scale=8.5,
    ):
        import torch

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator(self.device).manual_seed(int(seed))
        pipe = self._load_pipeline()
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        image = center_crop_single_object(image)
        image.save(output_path)
        return str(output_path)


class OpenAIGptImageProvider:
    name = "openai_gpt_image_1_5"

    def __init__(self, model_id=OPENAI_GPT_IMAGE_1_5_MODEL):
        self.model_id = model_id

    def _api_key(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI image generation")
        return api_key

    def _base_url(self):
        return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    def _size_string(self, width, height):
        if width >= height * 1.2:
            return "1536x1024"
        if height >= width * 1.2:
            return "1024x1536"
        return "1024x1024"

    def generate(
        self,
        prompt,
        negative_prompt,
        output_path,
        seed=1234,
        width=768,
        height=768,
        num_inference_steps=30,
        guidance_scale=8.5,
    ):
        from PIL import Image

        del seed, num_inference_steps, guidance_scale
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        merged_prompt = (
            f"{prompt}. Avoid: {negative_prompt}. "
            "Generate a single isolated object cutout with no environment, "
            "no floor, no pedestal, no support, and no cast shadow."
        )
        payload = {
            "model": self.model_id,
            "prompt": merged_prompt,
            "size": self._size_string(width, height),
            "quality": "medium",
            "background": "transparent",
            "output_format": "png",
        }
        req = urllib.request.Request(
            f"{self._base_url()}/images/generations",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8"))
        image_b64 = data["data"][0]["b64_json"]
        image_bytes = base64.b64decode(image_b64)
        with output_path.open("wb") as handle:
            handle.write(image_bytes)

        image = Image.open(output_path)
        image = center_crop_single_object(image)
        image.save(output_path)
        return str(output_path)



class SiliconFlowQwenImageProvider:
    name = "siliconflow_qwen_image"

    def __init__(self, model_id=SILICONFLOW_QWEN_IMAGE_MODEL):
        self.model_id = model_id

    def _api_key(self):
        api_key = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("SF_API_KEY")
        if not api_key:
            raise RuntimeError(
                "SILICONFLOW_API_KEY (or SF_API_KEY) is required for SiliconFlow image generation"
            )
        return api_key

    def _base_url(self):
        return os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")

    def _image_size(self, width, height):
        ratio = float(width) / float(max(height, 1))
        if ratio >= 1.45:
            return "1664x928"
        if ratio <= 0.70:
            return "928x1664"
        if ratio >= 1.20:
            return "1472x1140"
        if ratio <= 0.84:
            return "1140x1472"
        return "1328x1328"

    def generate(
        self,
        prompt,
        negative_prompt,
        output_path,
        seed=1234,
        width=768,
        height=768,
        num_inference_steps=30,
        guidance_scale=8.5,
    ):
        from PIL import Image

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image_size": self._image_size(width, height),
            "batch_size": 1,
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(guidance_scale),
            "seed": int(seed),
        }
        req = urllib.request.Request(
            f"{self._base_url()}/images/generations",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8"))

        image_url = None
        if "images" in data and data["images"]:
            image_url = data["images"][0].get("url")
        if image_url is None and "data" in data and data["data"]:
            image_url = data["data"][0].get("url")
        if image_url is None:
            raise RuntimeError(f"SiliconFlow response missing image url: {data}")

        urllib.request.urlretrieve(image_url, output_path)
        image = Image.open(output_path)
        image = center_crop_single_object(image)
        image.save(output_path)
        return str(output_path)


def center_crop_single_object(image, padding=28, threshold=245):
    """Crop away empty border so Hunyuan sees one large centered part.

    SDXL often leaves the object small on a white/light background. A simple
    foreground crop is intentionally conservative: if detection fails, return
    the original image unchanged.
    """
    try:
        from PIL import Image, ImageChops

        if "A" in image.getbands():
            rgba = image.convert("RGBA")
            alpha = rgba.getchannel("A")
            bbox = alpha.getbbox()
            if bbox is not None:
                left, upper, right, lower = bbox
                left = max(0, left - padding)
                upper = max(0, upper - padding)
                right = min(rgba.width, right + padding)
                lower = min(rgba.height, lower + padding)
                cropped = rgba.crop((left, upper, right, lower))
                canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 0))
                cropped.thumbnail((rgba.width - 2 * padding, rgba.height - 2 * padding))
                x = (rgba.width - cropped.width) // 2
                y = (rgba.height - cropped.height) // 2
                canvas.paste(cropped, (x, y), cropped)
                return canvas

        rgb = image.convert("RGB")
        background = Image.new("RGB", rgb.size, rgb.getpixel((0, 0)))
        diff = ImageChops.difference(rgb, background).convert("L")
        mask = diff.point(lambda value: 255 if value > threshold else 0)
        bbox = mask.getbbox()
        if bbox is None:
            return image
        left, upper, right, lower = bbox
        left = max(0, left - padding)
        upper = max(0, upper - padding)
        right = min(rgb.width, right + padding)
        lower = min(rgb.height, lower + padding)
        cropped = rgb.crop((left, upper, right, lower))
        canvas = Image.new("RGB", rgb.size, rgb.getpixel((0, 0)))
        cropped.thumbnail((rgb.width - 2 * padding, rgb.height - 2 * padding))
        x = (rgb.width - cropped.width) // 2
        y = (rgb.height - cropped.height) // 2
        canvas.paste(cropped, (x, y))
        return canvas
    except Exception:
        return image


def provider_from_name(name):
    if name == "flux_schnell":
        return FluxSchnellProvider()
    if name == "sdxl":
        return SdxlProvider()
    if name in ("openai_gpt_image_1_5", "gpt-image-1.5", "gpt_image_1_5"):
        return OpenAIGptImageProvider()
    if name in ("siliconflow_qwen_image", "qwen_image", "Qwen/Qwen-Image", "qwen/qwen-image"):
        return SiliconFlowQwenImageProvider()
    raise ValueError("supported providers: flux_schnell, sdxl, openai_gpt_image_1_5, siliconflow_qwen_image")


def write_t2i_metadata(path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return str(path)
