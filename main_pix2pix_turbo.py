import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F

from src.pix2pix_turbo import Pix2Pix_Turbo
from src.image_prep import canny_from_pil

if __name__ == "__main__":

    # TODO: The directory to save the output.
    output_folder = "outputs"
    # TODO: Name of the pretrained model to be used.
    model_name = "edge_to_image"
    # model_name = "sketch_to_image_stochastic"
    # TODO: Path to a local model state dict to be used.
    model_path = ""
    # TODO: The image preparation method.
    image_prep = "resize_512x512"
    # TODO: The canny edge detection low threshold.
    low_threshold = 75
    # TODO: The canny edge detection high threshold.
    high_threshold = 175
    # TODO: The sketch interpolation guidance amount.
    gamma = 0.4
    # TODO: The random seed to be used.
    seed = 42
    # TODO: The prompt to be used, also useful as caption. It is required when loading a custom model_path.
    prompt = "A rainy day in traffic."
    # TODO: The image source folder to translate from. Should be adjusted to the model name.
    input_folder = "inputs"
    subset_folder = "synthetic_images"
    image_files = [os.path.join(dp, f) for (dp, dn, fn) in os.walk(rf"{input_folder}\{subset_folder}") for f in fn]

    # initialize the model
    model = Pix2Pix_Turbo(pretrained_name=model_name, pretrained_path=model_path)
    model.set_eval()

    for image_file in image_files:
        # make sure that the input image is a multiple of 8
        input_image = Image.open(image_file).convert('RGB')
        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        caption = prompt.replace(' ', '_').replace('.', '')
        name = f'[Pix2Pix_Turbo]_[{caption}]_{os.path.basename(image_file)}'

        # translate the image
        with torch.no_grad():
            if model_name == 'edge_to_image':
                canny = canny_from_pil(input_image, low_threshold, high_threshold)
                canny_viz_inv = Image.fromarray(255 - np.array(canny))
                canny_viz_inv.save(os.path.join(output_folder, name.replace('.png', '_canny.png')))
                c_t = F.to_tensor(canny).unsqueeze(0).cuda()
                output_image = model(c_t, prompt)

            elif model_name == 'sketch_to_image_stochastic':
                image_t = F.to_tensor(input_image) < 0.5
                c_t = image_t.unsqueeze(0).cuda().float()
                torch.manual_seed(seed)
                B, C, H, W = c_t.shape
                noise = torch.randn((1, 4, H // 8, W // 8), device=c_t.device)
                output_image = model(c_t, prompt, deterministic=False, r=gamma, noise_map=noise)

            else:
                c_t = F.to_tensor(input_image).unsqueeze(0).cuda()
                output_image = model(c_t, prompt)

            output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)

        # save the output image
        output_pil.save(os.path.join(output_folder, name))
