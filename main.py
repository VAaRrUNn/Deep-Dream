
import argparse

from deepDream.deepdream_core import main_fn
def _main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cpu",
                        help="device to perform the computation on")

    # some issues in colab
    # parser.add_argument("--video", action="store_true",
    #                     help="convert the generated images to video")

    parser.add_argument("--config", type=str, default="default",
                        help="the configuration YAML file. It should be present inside config/")

    parser.add_argument("--image", type = str, default = None,
                        help="image path to dream on")
    
    args = parser.parse_args()
    image = args.image 
    device = args.device
    video = True
    config_name = args.config
    print(type(image))
    main_fn(image, 
            device, 
            video,
            config_name)
    # main_fn(image=args.image,
    #      device=args.device,
    #      video=True,
    #      config_name=args.config)


if __name__ == "__main__":
    _main()
