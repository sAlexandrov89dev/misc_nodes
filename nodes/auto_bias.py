import torch


class AutoBias:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bias": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "description": "Target midtone for bias adjustment",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "description": "Strength",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "misc_nodes/image"

    def process(self, image, bias=0.5, strength=1.0):
        # image is BHWC, RGB
        lum = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    
        min_val = lum.min()
        max_val = lum.max()
        av_val = lum.mean()
        range_val = max_val - min_val
    
        b = (av_val - min_val) / range_val
        t = (lum - min_val) / range_val
    
        # curve targeting bias instead of 0.5
        a = (b - bias) / (bias * (1.0 - b) + 1e-8)
        n = t / (a * (1.0 - t) + 1.0)
    
        # blend
        new_lum = torch.lerp(t, n, strength) * range_val + min_val
    
        # ratio transfer
        ratio = new_lum / (lum + 1e-8)
        result = image * ratio.unsqueeze(-1)
    
        return (result.clamp(0, 1), )


NODE_CLASS_MAPPINGS = {"AutoBias": AutoBias}
NODE_DISPLAY_NAME_MAPPINGS = {"AutoBias": "Auto Bias"}
