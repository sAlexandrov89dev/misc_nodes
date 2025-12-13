import torch
import torch.nn.functional as F


class EdgeAwareBlur:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "blur_amount": ("INT", {"default": 10, "min": 1, "max": 100, "display": "slider"}),
            "mask_expand": ("INT", {"default": 5, "min": 0, "max": 50, "display": "slider"}),
            "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("blurred", "edge_mask")
    FUNCTION = "blur"
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

    def blur(self, image, blur_amount, mask_expand, intensity):
        img = image.mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)
        original = img.clone()
        
        # Detect edges
        gx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
        gy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
        gx = F.pad(gx, [0, 1, 0, 0], mode='replicate')
        gy = F.pad(gy, [0, 0, 0, 1], mode='replicate')
        edge_mask = (gx + gy) / 2.0
        
        # Expand mask
        if mask_expand > 0:
            edge_mask = self.gaussian_blur(edge_mask, mask_expand)
        
        # Normalize and apply intensity
        edge_mask = edge_mask / (edge_mask.max() + 1e-8)
        edge_mask = edge_mask * intensity
        
        # Small radius, more iterations = smoother result
        radius = 2
        iterations = blur_amount
        
        result = original.clone()
        for _ in range(iterations):
            blurred = self.gaussian_blur(result, radius)
            result = result * (1 - edge_mask) + blurred * edge_mask
        
        result = result.permute(0, 2, 3, 1).expand(-1, -1, -1, 3)
        edge_mask = edge_mask.squeeze(1)
        
        return (result, edge_mask)


NODE_CLASS_MAPPINGS = {"EdgeAwareBlur": EdgeAwareBlur}
NODE_DISPLAY_NAME_MAPPINGS = {"EdgeAwareBlur": "Edge Aware Blur"}