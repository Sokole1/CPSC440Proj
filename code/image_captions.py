from load_dm import get_imagenet_dm_conf
from dataset import get_dataset
from utils import *
import torch
import torchvision
from tqdm.auto import tqdm
import random
from archs import get_archs, IMAGENET_MODEL
from advertorch.attacks import LinfPGDAttack
import matplotlib.pylab as plt
import time
import glob
import torchvision.models as models
from torch.utils.data import Subset, Dataset

from attack_tools import *
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont

import torch.optim as optim
from pred import d as d_dict


class Region_Denoised_Classifier(torch.nn.Module):
    def __init__(self, diffusion, model, classifier, t, mask):
        super().__init__()
        self.diffusion = diffusion
        self.model = model
        self.classifier = classifier
        self.t = t
        self.mask = mask

    def sdedit(self, x, t, to_01=True, mask=None):
        # assume the input is 0-1
        x = x * 2 - 1
        t = torch.full((x.shape[0],), t).long().to(x.device)
        x_t = self.diffusion.q_sample(x, t)
        sample = x_t
        indices = list(range(t + 1))[::-1]

        # visualize
        l_sample = []
        l_predxstart = []

        for i in indices:
            t_tensor = torch.full((x.shape[0],), i).long().to(x.device)
            out = self.diffusion.ddim_sample(self.model, sample, t_tensor)
            sample = out["sample"]

            l_sample.append(out["sample"])
            l_predxstart.append(out["pred_xstart"])

            if i > 0:
                sample = sample * mask + self.diffusion.q_sample(x, t_tensor - 1) * (
                    1 - mask
                )
            else:
                sample = sample * mask + x * (1 - mask)

        # visualize
        si(torch.cat(l_sample), "l_sample.png", to_01=1)
        si(torch.cat(l_predxstart), "l_pxstart.png", to_01=1)

        # the output of diffusion model is [-1, 1], should be transformed to [0, 1]
        if to_01:
            sample = (sample + 1) / 2

        return sample

    def forward(self, x):

        assert self.mask is not None

        out = self.sdedit(x, self.t, True, self.mask)  # [0, 1]
        out = self.classifier(out)
        return out


# ADD THIS IN, WAY TOO SLOW
@torch.no_grad()
def generate_x_adv_denoised_v2(
    x, y, diffusion, model, classifier, pgd_conf, device, t, mask
):
    net = Region_Denoised_Classifier(diffusion, model, classifier, t, mask)

    net = wrapper(net, x * (1 - mask), mask)
    delta = torch.zeros(x.shape).to(x.device)
    # delta.requires_grad_()

    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    eps = pgd_conf["eps"]
    alpha = pgd_conf["alpha"]
    iter = pgd_conf["iter"]

    for pgd_iter_id in range(iter):
        x_diff = net.net.sdedit(x + delta, t, mask=mask).detach()
        x_diff.requires_grad_()
        with torch.enable_grad():
            loss = loss_fn(classifier(x_diff), y)
            loss.backward()
            grad_sign = x_diff.grad.data.sign()

        delta += grad_sign * alpha

        delta = torch.clamp(delta, -eps, eps)

    print("Done")

    x_adv = torch.clamp(x + delta, 0, 1)
    return x_adv.detach()


def style_transfer(x, x_refer, mask, content_w, style_w, num_iters=300):
    model, style_losses, content_losses = get_style_model_and_losses(
        x_refer, x, style_w, content_w
    )

    x = x.clone()
    input_param = nn.Parameter(x)

    # optimizer =  optim.SGD([input_param], lr=0.01, momentum=0.9)
    optimizer = optim.Adam(
        [input_param],
        lr=0.01,
    )

    run = [0]
    while run[0] < num_iters:
        def closure():
            input_param.data.clamp_(0, 1)

            optimizer.zero_grad()
            input_param_new = input_param
            model(input_param_new)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.backward()
            for cl in content_losses:
                content_score += cl.backward()

            run[0] += 1
            if run[0] % 10 == 0:
                print("run {}:".format(run))
                print(
                    "Style Loss : {:4f} Content Loss: {:4f}".format(
                        style_score.item(), content_score.item()
                    )
                )
                print()

            return style_score + content_score

        optimizer.step(closure)

    input_param.data.clamp_(0, 1)
    return input_param


def add_caption(image, caption, color="white", font_size=20):
    caption = caption.split(",")[0]  # Just get the first one
    # Assume the input might be a single image in a batch-like format (1xCxHxW)
    was_tensor = isinstance(image, torch.Tensor)

    # If it's a tensor, check if it has a batch dimension
    if was_tensor:
        if image.ndim == 4 and image.shape[0] == 1:  # Check for single image batch
            image = image.squeeze(0)  # Remove the batch dimension
        image = transforms.ToPILImage()(image)

    # Create a font object
    font = ImageFont.load_default(size=font_size)
    draw = ImageDraw.Draw(image)
    # text_width, text_height = draw.textsize(caption, font=font)
    text_width, text_height = (font_size, font_size)

    # Create new image with space for caption
    new_image_height = image.height + text_height + 10  # 10 pixels padding
    new_image = Image.new("RGB", (image.width, new_image_height), "black")
    new_image.paste(image, (0, 0))

    # Draw the caption on the new image
    # text_position = ((image.width - text_width) // 2, image.height + 5)  # Center the text
    text_position = (5, image.height + 5)  # Center the text
    draw = ImageDraw.Draw(new_image)
    draw.text(text_position, caption, font=font, fill=color)

    # Convert back to tensor if the original input was a tensor
    if was_tensor:
        transform = transforms.ToTensor()
        new_image = transform(new_image)

    return new_image


def Attack_Region_Style(
    exp_name, x, mask, style_refer, classifier, device, respace, t, eps=16, iter=10, name="attack_style_new"
):
    pgd_conf = gen_pgd_confs(eps=eps, alpha=1, iter=iter, input_range=(0, 1))
    save_path = f"vis/{name}/{classifier}_eps{eps}_iter{iter}_{respace}_t{t}/"
    mp(save_path)
    save_path = f"vis/{name}/{classifier}_eps{eps}_iter{iter}_{respace}_t{t}/"
    mp(save_path)

    classifier = get_archs(classifier, "imagenet")
    classifier = classifier.to(device)
    classifier.eval()
    model, diffusion = get_imagenet_dm_conf(device=device, respace=respace)

    c = 0
    a = 0

    # ========== lamp =========
    # if exp_name == "lamp":
        # x = load_png(f"../data/advcam_dataset/other_imgs/img/lamp.jpg", 224)[
    #         None, ...
    #     ].to(device)
    #     mask = load_png(f"../data/advcam_dataset/other_imgs/seg/lamp.jpg", 224)[
    #         None, ...
    #     ].to(device)
    #     style_refer = load_png(
    #         f"../data/advcam_dataset/other_imgs/img/lamp-style.jpg", 224
    #     )[None, ...].to(device)

    # customize your own style with a different exp_name


    mask = (mask > 0).float()  # 1 means umasked, 0 means dont need to modify

    # do style transfer first
    # Original was 1000 iterations
    x_s = style_transfer(x, style_refer, mask, content_w=1, style_w=4000, num_iters=100)
    si(
        torch.cat([x, mask, style_refer, x_s * mask + x * (1 - mask)], -1),
        save_path + f"/{exp_name}_style_trans.png",
    )

    y_pred = classifier(x).argmax(1)  # original prediction
    x_s = x_s * mask + x * (1 - mask)
    x_s = x_s.detach()

    # generate DIFF-PGD Samples
    x_adv_diff_region = generate_x_adv_denoised_v2(
        x_s, y_pred, diffusion, model, classifier, pgd_conf, device, t, mask
    )

    # get purified sample
    with torch.no_grad():
        net = Region_Denoised_Classifier(diffusion, model, classifier, t, mask)
        x_adv_diff_p_region = net.sdedit(x_adv_diff_region, t, True, mask)

    target_pred = classifier(style_refer).argmax(1).item()
    x_s_pred = classifier(x_s).argmax(1).item()
    x_adv_no_purify_pred = classifier(x_adv_diff_region).argmax(1).item()
    y_final = classifier(x_adv_diff_p_region).argmax(1).item()

    print(
        f"Results for image {exp_name}:\n \
            Original = {y_pred.item()} : {d_dict[y_pred.item()]}\n  \
            Adv Region = {x_adv_no_purify_pred} : {d_dict[x_adv_no_purify_pred]}\n  \
            Adv Purified Region = {y_final} : {d_dict[y_final]}"
    )

    print("SAVING TO", save_path + f"/{exp_name}_final{y_final}.png")
    si(
        torch.cat(
            [
                torch.cat(
                    [
                        add_caption(x, f"O: {d_dict[y_pred.item()]}"),
                        #    add_caption(style_refer, f"T: {d_dict[target_pred]}"),
                        add_caption(style_refer, "Target (T)"),
                        add_caption(x_s, f"ST: {d_dict[x_s_pred]}"),
                        add_caption(
                            x_adv_diff_region,
                            f"Diff-PGD: {d_dict[x_adv_no_purify_pred]}",
                        ),
                        add_caption(
                            x_adv_diff_p_region, f"Purified: {d_dict[y_final]}"
                        ),
                        add_caption(mask, "mask"),
                    ],
                    -1,
                ),
                10
                * torch.cat(
                    [
                        add_caption(x - x, ""),
                        add_caption(style_refer - style_refer, ""),
                        add_caption(x_s - x, "Difference with O"),
                        add_caption(x_adv_diff_region - x_s, "Difference from ST"),
                        add_caption(
                            x_adv_diff_p_region - x_adv_diff_region,
                            "Difference from Diff-PGD",
                        ),
                        add_caption(mask - mask, ""),
                    ],
                    -1,
                ),
            ],
            -2,
        ),
        save_path + f"/{exp_name}_final{y_final}.png",
    )



import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

def show_first_image(dataset: Dataset):
    """Display the first image from a PyTorch dataset."""
    # Convert the first dataset item to a PIL image for display
    image, label = dataset[0]  # Assuming dataset[0] retrieves the first (image, label) tuple

    # Since the image might be a tensor, we need to convert it to a PIL Image for displaying
    image = ToPILImage()(image)

    # Display the image
    plt.imshow(image)
    plt.title(f'Label: {label}')
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()

# def show_all_images(dataset: Dataset):
#     """Display all images from a PyTorch dataset."""
#     fig = plt.figure(figsize=(10, 10))  # Adjust the figure size as needed

#     # Loop through all the images in the dataset
#     for i, (image, label) in enumerate(dataset):
#         ax = fig.add_subplot(1, len(dataset), i + 1)  # Adjust subplot grid parameters as necessary
#         ax.set_title(f'Label: {label}')
#         ax.axis('off')  # Turn off axis numbers and ticks

#         # Convert tensor image to PIL for displaying
#         image = ToPILImage()(image)
#         ax.imshow(image)

#     plt.show()
