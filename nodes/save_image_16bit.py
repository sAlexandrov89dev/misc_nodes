import numpy as np
from PIL import Image
import os
import folder_paths


class SaveImage16Bit:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "filename": ("STRING", {"default": "image_16bit"}),
            "exact_name": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "misc_nodes/image"

    def save(self, image, filename, exact_name):
        results = []
        counter = 0

        for img in image:
            if exact_name:
                final_name = f"{filename}.png" if not filename.endswith('.png') else filename
            else:
                while os.path.exists(os.path.join(self.output_dir, f"{filename}_{counter:05d}.png")):
                    counter += 1
                final_name = f"{filename}_{counter:05d}.png"
                counter += 1

            filepath = os.path.join(self.output_dir, final_name)
            gray = np.clip(img.cpu().numpy().mean(axis=-1), 0, 1)
            Image.fromarray((gray * 65535).astype(np.uint16), mode='I;16').save(filepath)
            results.append({"filename": final_name, "subfolder": "", "type": "output"})

        return {"ui": {"images": results}}


NODE_CLASS_MAPPINGS = {"SaveImage16Bit": SaveImage16Bit}
NODE_DISPLAY_NAME_MAPPINGS = {"SaveImage16Bit": "Save Image 16-bit PNG"}