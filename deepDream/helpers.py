from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import os

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

denormalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])


def image_converter(im):

    # move the image to cpu
    im_copy = im.cpu()

    # for plt.imshow() the channel-dimension is the last
    # therefore use transpose to permute axes
    im_copy = denormalize(im_copy.clone().detach()).numpy()
    im_copy = im_copy.transpose(1, 2, 0)

    # clip negative values as plt.imshow() only accepts
    # floating values in range [0,1] and integers in range [0,255]
    im_copy = im_copy.clip(0, 1)

    return im_copy


def save_image(image_array, image_name, image_path):
    save_path = Path(image_path) / Path(image_name)

    if not os.path.exists(image_path):
        os.makedirs(image_path)

    image = Image.fromarray((image_array * 255).astype(np.uint8))
    image.save(save_path)

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


def convert_to_video(images_path,
                     output_path,
                     initial_resolution,
                     final_resolution,
                     ext="png",
                     repeat_frames = 10,
                     steps=10,
                     fps = 10):
    
    print(f"Converting images to video")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, final_resolution)

    resolutions = [(initial_resolution[0] + (final_resolution[0] - initial_resolution[0]) * i // steps,
                    initial_resolution[1] + (final_resolution[1] - initial_resolution[1]) * i // steps)
                   for i in range(steps)]
    

    for res in resolutions:
        # Iterate through each image file in the folder
        for image_file in os.listdir(images_path):

            image_path = os.path.join(images_path, image_file)

            # Read the image
            image = cv2.imread(image_path)

            # Resize image to current resolution step
            resized_image = cv2.resize(image, res)

            # Write the resized image to the video
            out.write(resized_image)

    out.release()
    print("Video saved successfully!")
    cv2.destroyAllWindows()