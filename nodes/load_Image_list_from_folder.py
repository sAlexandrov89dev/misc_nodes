import os
from PIL import Image
import torch
import numpy as np
from aiohttp import web
from server import PromptServer

def select_folder_dialog():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder = filedialog.askdirectory()
    root.destroy()
    return folder

@PromptServer.instance.routes.get("/select_folder")
async def select_folder(request):
    import asyncio
    path = await asyncio.get_event_loop().run_in_executor(None, select_folder_dialog)
    return web.json_response({"path": path or ""})

class LoadImagesFromFolder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "folder": ("STRING", {"default": ""}),
            "limit": ("INT", {"default": 0, "min": 0, "max": 10000}),
        }}
    
    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "paths")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "load_images"
    
    def load_images(self, folder, limit):
        extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        image_paths = []
        
        for root, _, files in os.walk(folder):
            for f in files:
                if os.path.splitext(f)[1].lower() in extensions:
                    image_paths.append(os.path.join(root, f))
        
        image_paths.sort()
        
        if limit > 0:
            image_paths = image_paths[:limit]
        
        images = []
        for img_path in image_paths:
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img).astype(np.float32) / 255.0
            images.append(torch.from_numpy(img_array)[None,])
        
        return (images, image_paths)

NODE_CLASS_MAPPINGS = {"LoadImagesFromFolder": LoadImagesFromFolder}
NODE_DISPLAY_NAME_MAPPINGS = {"LoadImagesFromFolder": "Load Images From Folder"}