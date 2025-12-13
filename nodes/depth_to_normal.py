import torch
import torch.nn.functional as F


class DepthToNormal:
    SOBEL_X = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).float().view(1,1,3,3) / 4.0
    SOBEL_Y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]]).float().view(1,1,3,3) / 4.0

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "depth_map": ("IMAGE",),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10000.0, "step": 0.5, "display": "number"}),
            "invert_depth": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normal_map",)
    FUNCTION = "convert"
    CATEGORY = "misc_nodes/image"

    def convert(self, depth_map, strength, invert_depth):
        depth = depth_map.mean(dim=-1, keepdim=True).permute(0,3,1,2)
        if invert_depth:
            depth = 1.0 - depth

        sobel_x = self.SOBEL_X.to(depth.device, depth.dtype)
        sobel_y = self.SOBEL_Y.to(depth.device, depth.dtype)

        gx = F.conv2d(depth, sobel_x, padding=1) * strength
        gy = F.conv2d(depth, sobel_y, padding=1) * strength

        normals = torch.cat([-gx, -gy, torch.ones_like(gx)], dim=1)
        normals = F.normalize(normals, dim=1)
        normals = (normals.permute(0,2,3,1) + 1.0) / 2.0
        return (normals,)


NODE_CLASS_MAPPINGS = {"DepthToNormal": DepthToNormal}
NODE_DISPLAY_NAME_MAPPINGS = {"DepthToNormal": "Depth to Normal Map"}