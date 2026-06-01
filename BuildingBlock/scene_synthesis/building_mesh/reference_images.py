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
QWEN_IMAGE_LOCAL_MODEL = "Qwen/Qwen-Image"




def should_preserve_full_texture(prompt, negative_prompt=None):
    payload = f"{prompt} {negative_prompt or ''}".lower()
    return any(
        token in payload
        for token in (
            "texture swatch",
            "tileable",
            "seamless",
            "material texture",
            "surface material",
            "facade elevation texture",
            "exterior facade elevation",
            "flat uv texture",
            "wall face texture",
        )
    )


def normalize_provider_output_image(image, prompt, negative_prompt=None):
    if should_preserve_full_texture(prompt, negative_prompt):
        return image.convert("RGB")
    return normalize_reference_asset_image(image)

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
        image = normalize_provider_output_image(image, prompt, negative_prompt)
        image.save(output_path)
        return str(output_path)


class QwenImageLocalProvider:
    name = "qwen_image_local"

    def __init__(self, model_id=QWEN_IMAGE_LOCAL_MODEL, device="cuda"):
        self.model_id = os.environ.get("QWEN_IMAGE_MODEL_ID", model_id)
        self.device = os.environ.get("QWEN_IMAGE_DEVICE", device)
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is None:
            import torch
            from diffusers import QwenImagePipeline

            pipe = QwenImagePipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            # These are no-ops in older pipeline variants when unsupported, but
            # help keep 1024+ aspect-ratio canvases stable on 48GB GPUs.
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            if os.environ.get("QWEN_IMAGE_NO_OFFLOAD", "").lower() in ("1", "true", "yes"):
                pipe = pipe.to(self.device)
            elif hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload(device=self.device)
            elif hasattr(pipe, "enable_sequential_cpu_offload"):
                pipe.enable_sequential_cpu_offload(device=self.device)
            else:
                pipe = pipe.to(self.device)
            self._pipeline = pipe
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
        guidance_scale=4.0,
    ):
        import torch

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pipe = self._load_pipeline()
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or " ",
            width=int(width),
            height=int(height),
            num_inference_steps=int(num_inference_steps),
            true_cfg_scale=float(guidance_scale),
            generator=generator,
        ).images[0]
        image = normalize_provider_output_image(image, prompt, negative_prompt)
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
        image = normalize_provider_output_image(image, prompt, negative_prompt)
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
        image = normalize_provider_output_image(image, prompt, negative_prompt)
        image.save(output_path)
        return str(output_path)


class ProceduralReferenceProvider:
    """Deterministic offline fallback reference images for pipeline continuity.

    This provider is not meant to compete with real T2I quality.  It creates a
    clean, isolated, aspect-ratio-aware architectural part silhouette so the
    downstream geometry/report stages can keep running when external image APIs
    are unavailable or out of quota.  Metadata records the provider name, making
    fallback outputs easy to filter in later analysis.
    """

    name = "procedural_reference"

    def _palette(self, prompt):
        text = (prompt or "").lower()
        if any(token in text for token in ("glass", "window", "clerestory", "transparent", "glazing")):
            return (110, 160, 185), (45, 85, 105), (205, 235, 245)
        if any(token in text for token in ("roof", "eave", "canopy", "parapet", "ridge")):
            return (132, 118, 104), (82, 72, 64), (210, 200, 188)
        if any(token in text for token in ("pipe", "riser", "column", "post", "spire", "shaft")):
            return (150, 150, 145), (78, 78, 74), (220, 220, 215)
        if any(token in text for token in ("door", "portal", "bay", "stall")):
            return (150, 120, 88), (86, 62, 42), (220, 196, 166)
        if any(token in text for token in ("stone", "concrete", "wall", "facade", "podium")):
            return (165, 158, 146), (92, 88, 82), (222, 218, 210)
        return (155, 150, 140), (84, 82, 78), (224, 222, 216)

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
        del negative_prompt, num_inference_steps, guidance_scale
        import random
        from PIL import Image, ImageDraw

        rng = random.Random(int(seed) + sum(ord(ch) for ch in str(prompt)[:200]))
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        width = int(width)
        height = int(height)
        image = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(image)

        fill, edge, accent = self._palette(prompt)
        margin_x = max(32, int(width * 0.10))
        margin_y = max(32, int(height * 0.12))
        x0, y0 = margin_x, margin_y
        x1, y1 = width - margin_x, height - margin_y
        text = (prompt or "").lower()

        if any(token in text for token in ("column", "post", "pipe", "riser", "spire", "shaft")):
            cx = width // 2
            part_w = max(18, min(int(width * 0.22), x1 - x0))
            draw.rounded_rectangle(
                [cx - part_w // 2, y0, cx + part_w // 2, y1],
                radius=max(6, part_w // 8),
                fill=fill,
                outline=edge,
                width=max(3, width // 220),
            )
            cap_h = max(12, int(height * 0.035))
            draw.rectangle([cx - part_w, y0 - cap_h // 2, cx + part_w, y0 + cap_h], fill=accent, outline=edge)
            draw.rectangle([cx - part_w, y1 - cap_h, cx + part_w, y1 + cap_h // 2], fill=accent, outline=edge)
        elif any(token in text for token in ("window", "glass", "glazing", "clerestory", "transparent")):
            draw.rounded_rectangle([x0, y0, x1, y1], radius=10, fill=accent, outline=edge, width=max(3, width // 220))
            panes = 4 if (x1 - x0) > (y1 - y0) else 2
            for i in range(1, panes):
                x = x0 + (x1 - x0) * i / panes
                draw.line([x, y0, x, y1], fill=edge, width=max(2, width // 300))
            draw.line([x0, (y0 + y1) / 2, x1, (y0 + y1) / 2], fill=edge, width=max(2, height // 300))
            for _ in range(3):
                sx = rng.randint(x0 + 10, max(x0 + 12, x1 - 30))
                draw.line([sx, y0 + 8, min(x1 - 8, sx + 60), y0 + 38], fill=(240, 250, 255), width=2)
        elif any(token in text for token in ("roof", "eave", "canopy", "ridge", "sawtooth")):
            if "sawtooth" in text:
                teeth = 4
                points = [(x0, y1)]
                step = (x1 - x0) / teeth
                for i in range(teeth):
                    points.extend([(x0 + i * step + step * 0.55, y0), (x0 + (i + 1) * step, y1)])
                points.append((x0, y1))
                draw.polygon(points, fill=fill, outline=edge)
            else:
                draw.polygon([(x0, y1), (width // 2, y0), (x1, y1)], fill=fill, outline=edge)
                draw.rectangle([x0, int(y1 - (y1 - y0) * 0.18), x1, y1], fill=accent, outline=edge)
        elif any(token in text for token in ("stair", "steps")):
            steps = 5
            for i in range(steps):
                yy1 = y1 - i * (y1 - y0) / steps
                yy0 = y1 - (i + 1) * (y1 - y0) / steps
                xx0 = x0 + i * (x1 - x0) / (steps * 2)
                draw.rectangle([xx0, yy0, x1, yy1], fill=fill if i % 2 else accent, outline=edge)
        else:
            draw.rounded_rectangle([x0, y0, x1, y1], radius=12, fill=fill, outline=edge, width=max(3, width // 220))
            # Generic architectural detail lines: horizontal courses plus a few
            # vertical divisions keep the image part-like without adding scene.
            for i in range(1, 5):
                y = y0 + (y1 - y0) * i / 5
                draw.line([x0 + 8, y, x1 - 8, y], fill=accent, width=max(1, height // 360))
            for i in range(1, 4):
                x = x0 + (x1 - x0) * i / 4
                draw.line([x, y0 + 12, x, y1 - 12], fill=edge, width=max(1, width // 420))

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


def normalize_reference_asset_image(image, padding=28, threshold=245):
    """Return a high-contrast white-background object reference for 3D stages.

    T2I providers often leave faint tinted borders, gray studio floors, or
    alpha/near-white artifacts.  Those artifacts can be interpreted by image-to-3D
    models as physical backing boards or bases.  This postprocess stays generic:
    crop the foreground, recenter it, and rebuild a clean white canvas.
    """
    try:
        from PIL import Image

        cropped = center_crop_single_object(image, padding=padding, threshold=threshold)
        if "A" in cropped.getbands():
            rgba = cropped.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            white.alpha_composite(rgba)
            return white.convert("RGB")
        return cropped.convert("RGB")
    except Exception:
        return image


def provider_from_name(name):
    if name == "flux_schnell":
        return FluxSchnellProvider()
    if name == "sdxl":
        return SdxlProvider()
    if name in ("qwen_image_local", "qwen_local", "Qwen/Qwen-Image-local"):
        return QwenImageLocalProvider()
    if name in ("openai_gpt_image_1_5", "gpt-image-1.5", "gpt_image_1_5"):
        return OpenAIGptImageProvider()
    if name in ("siliconflow_qwen_image", "qwen_image", "Qwen/Qwen-Image", "qwen/qwen-image"):
        return SiliconFlowQwenImageProvider()
    if name in ("procedural_reference", "procedural", "offline_reference"):
        return ProceduralReferenceProvider()
    raise ValueError(
        "supported providers: flux_schnell, sdxl, qwen_image_local, "
        "openai_gpt_image_1_5, siliconflow_qwen_image, procedural_reference"
    )


def write_t2i_metadata(path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return str(path)
