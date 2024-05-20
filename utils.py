import torch
import numpy as np
from sensor_msgs.msg import Image
from PIL import Image as PILImage
from typing import List, Tuple, Dict, Optional
from torchvision import transforms

def to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def load_model_para(model, config):
    checkpoint = torch.load(config["ckpt_path"], map_location='cuda:0')
    model.load_state_dict(checkpoint)
    return model

def msg_to_pil(msg: Image) -> PILImage.Image:
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, -1)
    pil_image = PILImage.fromarray(img)
    return pil_image

def transform_images(pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225]),
        ]
    )
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        pil_img = pil_img.resize(image_size) 
        pil_img = pil_img.convert("RGB")
        transf_img = transform_type(pil_img)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    print(len(transf_imgs))
    return torch.cat(transf_imgs, dim=1)

def waypoint_normalize(waypoint_ori, x_min, x_max, y_min, y_max):
    waypoint = waypoint_ori.clone()
    assert waypoint.shape[-1] == 2
    waypoint[...,0] = (waypoint[...,0]-x_min)/(x_max-x_min)
    waypoint[...,0] = waypoint[...,0]*2 - 1
    waypoint[...,1] = (waypoint[...,1]-y_min)/(y_max-y_min)
    waypoint[...,1] = waypoint[...,1]*2 - 1
    return waypoint

def waypoint_unnormalize(waypoint_ori, x_min, x_max, y_min, y_max):
    waypoint = waypoint_ori.clone()
    assert waypoint.shape[-1] == 2
    waypoint = (waypoint+1)/2
    
    waypoint[...,0] = waypoint[...,0]*(x_max-x_min)+x_min
    waypoint[...,1] = waypoint[...,1]*(y_max-y_min)+y_min
    return waypoint