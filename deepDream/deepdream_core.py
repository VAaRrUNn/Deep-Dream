import torch
import torch.nn as nn
import numpy as np
from torchvision import models
import sys
import cv2
import hydra
from PIL import Image

from deepDream.gradients import RGBgradients
from deepDream.helpers import image_converter, normalize, denormalize, save_image, convert_to_video, get_resolution, resize_image


def create_hook(name, activation):
    def hook(module, inp, out):
        activation[name] = out
    return hook


def grad_loss(img, gradLayer,  beta=1, device='cpu'):

    # move the gradLayer to cuda
    gradLayer.to(device)
    gradSq = gradLayer(img.unsqueeze(0))**2

    grad_loss = torch.pow(gradSq.mean(), beta/2)

    return grad_loss


def dream(model,
          image,
          activation,
          act_wt=0.5,
          upscaling_factor=1.5,
          upscaling_steps=1,
          optim_steps=20,
          device="cpu",
          image_save_path=None,
          gradLayer=None,
          neuron_index=11):
    print("Starting dreaming...")
    model.eval()
    image_index = 0
    for mag_epoch in range(upscaling_steps+1):
        optimizer = torch.optim.Adam(params=[image], lr=0.4)
        image = image.to(device)

        for opt_epoch in range(optim_steps):
            optimizer.zero_grad()
            model(image.unsqueeze(0))
            print("passed ;;")
            layer_out = activation['4a']
            rms = torch.pow((layer_out[0, neuron_index]**2).mean(), 0.5)
            # terminate if rms is nan
            if torch.isnan(rms):
                print('rms is nan, exiting')
                sys.exit()

            # pixel intensity
            pxl_inty = torch.pow((image**2).mean(), 0.5)
            # terminate if pxl_inty is nan
            if torch.isnan(pxl_inty):
                print('pixel intensity is nan, exiting')
                sys.exit()

            # image gradients
            im_grd = grad_loss(image, gradLayer=gradLayer,
                               beta=1, device=device)
            # terminate is im_grd is nan
            if torch.isnan(im_grd):
                print('image gradients are nan, exiting')
                sys.exit()

            loss = -act_wt*rms + pxl_inty + im_grd

            # print activation at the beginning of each mag_epoch
            loss.backward()
            optimizer.step()

            img = image_converter(image)
            image_index += 1
            image_name = str(image_index) + ".png"

            # saving the image in a temperory folder
            save_image(image_array=img,
                       image_name=image_name,
                       image_path=image_save_path)

        print(f"Done: {mag_epoch}/{upscaling_steps}")
        img = cv2.resize(img, dsize=(0, 0),
                         fx=upscaling_factor, fy=upscaling_factor).transpose(2, 0, 1)  # scale up and move the batch axis to be the first
        image = normalize(torch.from_numpy(img)).to(
            device).requires_grad_(True)

    return image


def main_fn(image=None, device=None, video=None, config_name="default"):
    # this is messing with the argparser from main.py file so instead loading manually
    # @hydra.main(version_base=None, config_path="../config", config_name=config_name, allow_unknown_args = True)
    def _main(cfg):
        nonlocal image, device
        activation = {}
        model = models.googlenet(pretrained=True)
        for param in model.parameters():
            param.requires_grad_(False)

        # choosing inception4e by default
        model.inception4e.register_forward_hook(create_hook('4a', activation))
        filter_x = np.array([[-3, 0, 3],
                            [-10, 0, 10],
                            [-3, 0, 3]])

        filter_y = filter_x.T
        grad_filters = np.array([filter_x, filter_y])

        gradLayer = RGBgradients(grad_filters)

        if image != None:
            try:
                pil_image = Image.open(image)
                np_image = np.array(pil_image)
                # np_image = resize_image(np_image, 28, 28)
                image = torch.from_numpy(np_image)
                image = image.to(torch.float32)
                image = torch.permute(image, (2, 0, 1))
                image = normalize(image)
                image.requires_grad_(True)
            except Exception as e:
                print(f"Error loading: {image}")
                image = None
                sys.exit()

        if image == None:
            img = np.single(np.random.uniform(0, 1, (3, cfg.image_dim.height,
                                                     cfg.image_dim.width)))
            im_tensor = normalize(torch.from_numpy(img)).requires_grad_(True)
            img_tensor = im_tensor
            image = img_tensor.detach().clone()
            image.requires_grad_(True)

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception as e:
            device = cfg.device

        print(f"Running on device {device}")
        model.to(device)
        initial_resolution = get_resolution(image)
        final_image = dream(model=model,
                            image=image,
                            activation=activation,
                            act_wt=cfg.hyperparameters.act_weight,
                            upscaling_factor=cfg.hyperparameters.upscaling_factor,
                            upscaling_steps=cfg.hyperparameters.upscaling_steps,
                            optim_steps=cfg.hyperparameters.optimization_steps,
                            device=device,
                            image_save_path=cfg.path.temp_image_path,
                            gradLayer=gradLayer,
                            neuron_index=cfg.hyperparameters.neuron_index)

        final_resolution = get_resolution(final_image)

        # converting images to videos
        try:
            if video:
                convert_to_video(images_path=cfg.path.temp_image_path,
                                 output_path=cfg.video.output_path,
                                 ext=cfg.video.ext,
                                 repeat_frames=cfg.video.repeat_frames,
                                 initial_resolution=initial_resolution,
                                 final_resolution=final_resolution)
        except Exception as e:
            print("Error in converting images to videos...")

    with hydra.initialize(version_base=None, config_path="../config", job_name="deepdream"):
        cfg = hydra.compose(config_name="default")
    _main(cfg)


if __name__ == "__main__":
    main()
