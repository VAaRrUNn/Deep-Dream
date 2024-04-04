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

    image = Image.fromarray(image_array)
    image.save(save_path)


def convert_to_video(images_path,
                     output_path,
                     ext="png",
                     width=28,
                     height=28,
                     repeat_frames = 10):

    images = []
    for f in os.listdir(images_path):
        if f.endswith(ext):
            images.append(f)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

    for image in images:

        image_path = os.path.join(images_path, image)
        frame = cv2.imread(image_path)

        for a in repeat_frames:
            out.write(frame)

    out.release()
    print("Video saved successfully!")
    cv2.destroyAllWindows()
