from deepDream.deepdream_core import main
import argparse


def _main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cpu",
                        help="device to perform the computation on")

    # some issues in colab
    # parser.add_argument("--video", action="store_true",
    #                     help="convert the generated images to video")

    parser.add_argument("--config", type=str, default="default",
                        help="the configuration YAML file. It should be present inside config/")

    parser.add_argument("-i", "--image", type = str, default = None,
                        help="image path to dream on")

    args = parser.parse_args()
    print(args)
    main(image=args.image,
         device=args.device,
         video=True,
         config_name=args.config)


if __name__ == "__main__":
    _main()
