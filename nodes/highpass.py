import torch
import torch.nn.functional as F


class Highpass:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "radius": ("INT", {"default": 10, "min": 1, "max": 100, "display": "slider"}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "display": "slider"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("highpass",)
    FUNCTION = "process"
    CATEGORY = "misc_nodes/image"

    def gaussian_blur(self, x, radius):
        if radius < 1:
            return x
        sigma = radius / 2.0
        size = radius * 2 + 1
        coords = torch.arange(size, device=x.device, dtype=x.dtype) - radius
        gauss = torch.exp(-coords**2 / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        x = F.conv2d(x, gauss.view(1, 1, -1, 1), padding=(radius, 0))
        x = F.conv2d(x, gauss.view(1, 1, 1, -1), padding=(0, radius))
        return x

    def process(self, image, radius, strength):
        img = image.mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)
        
        # Highpass = original - lowpass
        lowpass = self.gaussian_blur(img, radius)
        highpass = img - lowpass
        
        # Center at 0.5, apply strength
        result = 0.5 + highpass * strength
        result = torch.clamp(result, 0, 1)
        
        result = result.permute(0, 2, 3, 1).expand(-1, -1, -1, 3)
        return (result,)


NODE_CLASS_MAPPINGS = {"Highpass": Highpass}
NODE_DISPLAY_NAME_MAPPINGS = {"Highpass": "Highpass Filter"}