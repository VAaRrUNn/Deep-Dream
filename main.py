
import argparse

from deepDream.deepdream_core import main_fn
def _main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cpu",
                        help="device to perform the computation on")

    # some issues in colab
    parser.add_argument("--video", action="store_true",
                        help="convert the generated images to video")

    parser.add_argument("--config", type=str, default="default",
                        help="the configuration YAML file. It should be present inside config/")

    parser.add_argument("--image", type = str, default = None,
                        help="image path to dream on")
    
    parser.add_argument("--keep_folder", action="store_false",
                        help = "specifies to keep the image folder or not.")
    
    args = parser.parse_args()
    main_fn(image=args.image,
         device=args.device,
         video=args.video,
         config_name=args.config,
         keep_folder= args.keep_folder)


if __name__ == "__main__":
    _main()
