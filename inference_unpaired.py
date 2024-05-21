import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from src.cyclegan_turbo import CycleGAN_Turbo
from src.my_utils.training_utils import build_transform


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input_image', type=str, required=False, help='the path to the input image.')
    # parser.add_argument('--prompt', type=str, required=False, help='The prompt. Required for custom model_path.')
    # parser.add_argument('--model_name', type=str, default=None, help='The name of the pretrained model to be used.')
    # parser.add_argument('--model_path', type=str, default=None, help='The path to a local model state dict.')
    # parser.add_argument('--output_dir', type=str, default='output', help='The directory to save the output.')
    # parser.add_argument('--image_prep', type=str, default='resize_512x512', help='The image preparation method.')
    # parser.add_argument('--direction', type=str, default=None,
    #                     help='The direction of translation. None for pretrained models, a2b or b2a for custom paths.')
    # args = parser.parse_args()

    # # only one of model_name and model_path should be provided
    # if args.model_name is None and args.model_path is None:
    #     raise ValueError('Either model_name or model_path should be provided')
    #
    # if args.model_path is not None and args.prompt is None:
    #     raise ValueError('Prompt is required when loading a custom model_path.')
    #
    # if args.model_name is not None:
    #     assert args.prompt is None, 'Prompt is not required when loading a pretrained model.'
    #     assert args.direction is None, 'Direction is not required when loading a pretrained model.'

    # TODO: Name of the pretrained model to be used.
    # model_name = "clear_to_rainy"
    model_name = "day_to_night"
    # model_name = "night_to_day"
    # model_name = "rainy_to_clear"
    # TODO: The directory to save the output.
    output_folder = "outputs"
    # TODO: Path to a local model state dict to be used.
    # model_path = "checkpoints/clear2rainy.pkl"
    model_path = "checkpoints/day2night.pkl"
    # model_path = "checkpoints/night2day.pkl"
    # model_path = "checkpoints/rainy2clear.pkl"
    # TODO: The prompt to be used, also useful as caption. It is required when loading a custom model_path.
    prompt = None
    # TODO: The image preparation method.
    # image_prep = "no_resize"
    image_prep = "resize_512x512"
    # TODO: The direction of translation. None for pretrained models, a2b or b2a for custom paths.
    direction = None
    # TODO: The image source folder to translate from. Should be adjusted to the model name.
    input_folder = "inputs"
    subset_folder = "other_samples"
    # subset_folder = "clear_images"
    # subset_folder = "day_images"
    # subset_folder = "night_images"
    # subset_folder = "synthetic_images"
    image_files = [os.path.join(dp, f) for (dp, dn, fn) in os.walk(rf"{input_folder}\{subset_folder}") for f in fn]

    # initialize the model
    model = CycleGAN_Turbo(pretrained_name=model_name, pretrained_path=model_path)
    model.eval()
    model.unet.enable_xformers_memory_efficient_attention()

    T_val = build_transform(image_prep)

    for image_file in image_files:
        input_image = Image.open(image_file).convert('RGB')
        # translate the image
        with torch.no_grad():
            input_img = T_val(input_image)
            x_t = transforms.ToTensor()(input_img)
            x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).cuda()
            output = model(x_t, direction=direction, caption=prompt)

        output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
        resized_image = output_pil.resize((input_image.width, input_image.height), Image.LANCZOS)

        # Save the output image
        name = f"[CycleGAN_Turbo]_[{model_name}]_{os.path.basename(image_file)}"
        os.makedirs(output_folder, exist_ok=True)
        resized_image.save(os.path.join(output_folder, name))
