import os
import torch
from PIL import Image
from torchvision import transforms

from src.cyclegan_turbo import CycleGAN_Turbo
from src.my_utils.training_utils import build_transform

if __name__ == "__main__":

    # TODO: The directory to save the output.
    output_folder = "outputs"
    # TODO: Name of the pretrained model to be used.
    # model_name = "clear_to_rainy"
    model_name = "day_to_night"
    # model_name = "night_to_day"
    # model_name = "rainy_to_clear"
    # TODO: Path to a local model state dict to be used.
    # model_path = "checkpoints/clear2rainy.pkl"
    model_path = "checkpoints/day2night.pkl"
    # model_path = "checkpoints/night2day.pkl"
    # model_path = "checkpoints/rainy2clear.pkl"
    # TODO: The image preparation method.
    # image_prep = "no_resize"
    image_prep = "resize_512x512"
    # TODO: The direction of translation. None for pretrained models, a2b or b2a for custom paths.
    direction = None
    # TODO: The prompt to be used, also useful as caption. It is required when loading a custom model_path.
    prompt = ""
    # TODO: The image source folder to translate from. Should be adjusted to the model name.
    input_folder = "inputs"
    # subset_folder = "clear_images"
    subset_folder = "day_images"
    # subset_folder = "night_images"
    image_files = [os.path.join(dp, f) for (dp, dn, fn) in os.walk(rf"{input_folder}\{subset_folder}") for f in fn]
    # image_files = ["assets/examples/clear2rainy_input.png"]
    # image_files = ["assets/examples/day2night_input.png"]
    # image_files = ["assets/examples/night2day_input.png"]
    # image_files = ["assets/examples/rainy2clear_input.png"]

    # initialize the model
    model = CycleGAN_Turbo(pretrained_name=model_name, pretrained_path=model_path)
    model.eval()
    model.unet.enable_xformers_memory_efficient_attention()

    T_val = build_transform(image_prep)

    for image_file in image_files:
        image = Image.open(image_file).convert('RGB')
        # translate the image
        with torch.no_grad():
            input_img = T_val(image)
            x_t = transforms.ToTensor()(input_img)
            x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).cuda()
            output = model(x_t, direction=direction, caption=prompt)

        output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
        output_pil = output_pil.resize((image.width, image.height), Image.LANCZOS)

        # save the output image
        name = f"[CycleGAN_Turbo]_[{model_name}]_{os.path.basename(image_file)}"
        os.makedirs(output_folder, exist_ok=True)
        output_pil.save(os.path.join(output_folder, name))
