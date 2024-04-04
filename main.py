from deepDream.deepdream_core import main
import argparse


def _main():

    parser = argparse.ArgumentParser()

    parser.add_argument("device", type=str, default="cpu",
                        help="device to perform the computation on")

    parser.add_argument("--video", action="store_true",
                        help="convert the generated images to video")

    parser.add_argument("--config", type=str, default="default",
                        help="the configuration YAML file. It should be present inside config/")

    parser.add_argument("-i", "--image", action="store_true",
                        help="image path to dream on")

    args = parser.parse_args()

    main(image=args.image,
         device=args.device,
         video=args.video,
         config_name=args.config_name)


if __name__ == "__main__":
    _main()
