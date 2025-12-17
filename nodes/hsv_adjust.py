import torch


class HSVAdjust:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "hue": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0, "display": "slider"}),
            "saturation": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
            "brightness": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
            "colorize": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "adjust"
    CATEGORY = "misc_nodes/image"

    def rgb_to_hsv(self, rgb):
        r, g, b = rgb[:, :, :, 0], rgb[:, :, :, 1], rgb[:, :, :, 2]
        
        max_c, _ = rgb.max(dim=-1)
        min_c, _ = rgb.min(dim=-1)
        diff = max_c - min_c
        
        h = torch.zeros_like(max_c)
        mask = diff > 0
        
        r_max = (max_c == r) & mask
        g_max = (max_c == g) & mask
        b_max = (max_c == b) & mask
        
        h[r_max] = ((g[r_max] - b[r_max]) / diff[r_max]) % 6
        h[g_max] = ((b[g_max] - r[g_max]) / diff[g_max]) + 2
        h[b_max] = ((r[b_max] - g[b_max]) / diff[b_max]) + 4
        h = h / 6.0
        
        s = torch.zeros_like(max_c)
        s[max_c > 0] = diff[max_c > 0] / max_c[max_c > 0]
        
        v = max_c
        
        return torch.stack([h, s, v], dim=-1)

    def hsv_to_rgb(self, hsv):
        h, s, v = hsv[:, :, :, 0], hsv[:, :, :, 1], hsv[:, :, :, 2]
        
        h = h * 6.0
        i = h.floor()
        f = h - i
        
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        
        i = i.long() % 6
        
        rgb = torch.zeros_like(hsv)
        
        mask = i == 0
        rgb[mask] = torch.stack([v[mask], t[mask], p[mask]], dim=-1)
        mask = i == 1
        rgb[mask] = torch.stack([q[mask], v[mask], p[mask]], dim=-1)
        mask = i == 2
        rgb[mask] = torch.stack([p[mask], v[mask], t[mask]], dim=-1)
        mask = i == 3
        rgb[mask] = torch.stack([p[mask], q[mask], v[mask]], dim=-1)
        mask = i == 4
        rgb[mask] = torch.stack([t[mask], p[mask], v[mask]], dim=-1)
        mask = i == 5
        rgb[mask] = torch.stack([v[mask], p[mask], q[mask]], dim=-1)
        
        return rgb

    def adjust(self, image, hue, saturation, brightness, colorize):
        hsv = self.rgb_to_hsv(image)
        
        hue_shift = hue / 360.0
        sat_factor = saturation * 2.0
        
        if colorize:
            hsv[:, :, :, 0] = hue_shift
            hsv[:, :, :, 1] = sat_factor / 2.0
        else:
            hsv[:, :, :, 0] = (hsv[:, :, :, 0] + hue_shift) % 1.0
            hsv[:, :, :, 1] = torch.clamp(hsv[:, :, :, 1] * sat_factor, 0, 1)
        
        # Brightness: 0 = black, 0.5 = original, 1 = white
        v = hsv[:, :, :, 2]
        if brightness <= 0.5:
            # Multiply toward black
            v = v * (brightness * 2)
        else:
            # Blend toward white
            factor = (brightness - 0.5) * 2
            v = v + (1 - v) * factor
        hsv[:, :, :, 2] = v
        
        rgb = self.hsv_to_rgb(hsv)
        return (rgb,)


NODE_CLASS_MAPPINGS = {"HSVAdjust": HSVAdjust}
NODE_DISPLAY_NAME_MAPPINGS = {"HSVAdjust": "HSV Adjust"}