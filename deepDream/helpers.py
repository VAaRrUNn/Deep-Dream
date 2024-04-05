from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import os
import re
import torch

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

denormalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])


def image_converter(im):
    im_copy = im.clone().detach().cpu()
    im_copy = denormalize(im_copy).type(torch.uint8).permute(1, 2, 0).numpy()
    print(im_copy.shape)
    print(im_copy[0][0])
    # clip negative values as plt.imshow() only accepts
    # floating values in range [0,1] and integers in range [0,255]
    # im_copy = im_copy.clip(0, 1)
    return im_copy


def save_image(image_array, image_name, image_path):
    save_path = Path(image_path) / Path(image_name)

    if not os.path.exists(image_path):
        os.makedirs(image_path)

    print("going for conversion")

    image = Image.fromarray(image_array)
    print("after")
    # print(image[0][0])
    print(image)

    # image = Image.fromarray((image_array * 255).astype(np.uint8))
    image.save(save_path)
    print("fdone")

def get_resolution(image_tensor):
    if len(image_tensor.shape) == 3:
        height, width, _ = image_tensor.shape
    # Check if the tensor is 2-dimensional (assuming it represents a grayscale image)
    elif len(image_tensor.shape) == 2:
        height, width = image_tensor.shape
    else:
        raise ValueError("Invalid tensor shape. Expected 2 or 3 dimensions.")

    return (height, width)


def resize_image(image, width, height):
    # image = image.detach().cpu().numpy()
    image = cv2.resize(image, (width, height))

    return image

# def image_converter(im):
#     im_copy = im.cpu()
#     im_copy = denormalize(im_copy.clone().detach()).numpy()
#     im_copy = im_copy.transpose(1,2,0)
    
#     # clip negative values as plt.imshow() only accepts 
#     # floating values in range [0,1] and integers in range [0,255]
#     im_copy = im_copy.clip(0, 1) 
    
#     return im_copy

def extract_image_number(image_name):
    pattern = r'(\d+)\.png'
    match = re.search(pattern, image_name)
    if match:
        return match.group(1)
    else:
        return None

def convert_to_video(images_path,
                     output_path,
                     resolution,
                     ext="png",
                     repeat_frames = 1,
                     fps = 10):
        
        print(f"Converting images to video")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, resolution)

        # Getting a list of images from the folder.
        image_files = [str(f) for f in Path(images_path).glob('*.png')] 

        # sort the images
        sorted_images = dict()
        for image in image_files:
            idx = extract_image_number(image)
            if idx:
                sorted_images[int(idx)] = image
        
        del image_files
        sorted_images = dict(sorted(sorted_images.items()))
        # print(sorted_images)
        for image in sorted_images.values():
            # print(image)
            image = cv2.imread(image)
            resized_image = cv2.resize(image, resolution)
            for _ in range(repeat_frames):
                 video_writer.write(resized_image)

        video_writer.release()
        cv2.destroyAllWindows()
        print(f'Video saved to {output_path}')
