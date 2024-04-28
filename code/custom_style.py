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
import cv2
import torch
import numpy as np
from torchvision.transforms import ToPILImage, ToTensor

from attack_global import Denoised_Classifier
from attack_global import generate_x_adv_denoised_v2 as generate_x_adv_denoised_global

def Attack_Global(
    x,
    y,
    classifier,
    device,
    respace,
    t,
    eps=16,
    iter=10,
    name="attack_global",
    alpha=2,
    version="v2",
    skip=200,
):
    pgd_conf = gen_pgd_confs(eps=eps, alpha=alpha, iter=iter, input_range=(0, 1))
    save_path = f"vis/{name}_{version}/{classifier}_eps{eps}_iter{iter}_{respace}_t{t}/"
    mp(save_path)

    classifier = get_archs(classifier, "imagenet")
    classifier = classifier.to(device)
    classifier.eval()

    dataset = get_dataset("imagenet", split="test")

    # dataset = Subset(dataset, indices=np.arange(100))  # for testing, just use 2.
    model, diffusion = get_imagenet_dm_conf(
        device=device, respace=respace, model_path="../ckpt/256x256_diffusion_uncond.pt"
    )

    # x = x[None,].to(device)
    # y = torch.tensor(y)[None,].to(device)

    y_pred = classifier(x).argmax(1)  # original prediction

    if version == "v2":
        x_adv = generate_x_adv_denoised_global(
            x, y_pred, diffusion, model, classifier, pgd_conf, device, t
        )
    
    with torch.no_grad():
        net = Denoised_Classifier(diffusion, model, classifier, t)
        pred_x0 = net.sdedit(x_adv, t)


    # si(torch.cat([x, x_adv, pred_x0], -1), save_path + f"{i}.png")

    x_adv_pred = d_dict[classifier(x_adv).argmax(1).item()]
    x_purify_pred = d_dict[classifier(pred_x0).argmax(1).item()]

    print(f"Results for image:\n Original = {y_pred.item()} : {d_dict[y_pred.item()]}\n Adv = {classifier(x_adv).argmax(1).item()} : {d_dict[classifier(x_adv).argmax(1).item()]}\n Adv_diff = {classifier(pred_x0).argmax(1).item()} : {d_dict[classifier(pred_x0).argmax(1).item()]}")
    print(
        "Global: Corresponding labels for: Original, Adv, Adv_diff for image:", 
        d_dict[y_pred.item()],
        x_adv_pred,
        x_purify_pred
    )

    images_tensor =  torch.cat(
        [
            torch.cat(
                [
                    add_caption(x, f"O: {d_dict[y_pred.item()]}"),
                    add_caption(x - x, ""), # Pad so size matches
                    add_caption(x - x, ""),
                    add_caption(
                        x_adv,
                        f"Diff-PGD: {x_adv_pred}",
                    ),
                    add_caption(
                        pred_x0, f"Purified: {x_purify_pred}"
                    ),
                    add_caption(x - x, ""),
                ],
                -1,
            ),
            10
            * torch.cat(
                [
                    add_caption(x - x, ""),
                    add_caption(x - x, ""),
                    add_caption(x - x, ""),
                    add_caption(x_adv - x, "Difference from O"),
                    add_caption( pred_x0 - x_adv,
                        "Difference from Diff-PGD"),
                    add_caption(x - x, ""),
                ],
                -1,
            ),
        ],
        -2,
    )

    res_adv = 1 if d_dict[y_pred.item()] != d_dict[classifier(x_adv).argmax(1).item()] else 0
    res_pur = 1 if d_dict[y_pred.item()] != d_dict[classifier(pred_x0).argmax(1).item()] else 0
    return res_adv, res_pur, images_tensor


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

    return x_adv * mask + x * (1 - mask)


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
            # if run[0] % 10 == 0:
            #     print("run {}:".format(run))
            #     print(
            #         "Style Loss : {:4f} Content Loss: {:4f}".format(
            #             style_score.item(), content_score.item()
            #         )
            #     )
            #     print()

            return style_score + content_score

        optimizer.step(closure)

    input_param.data.clamp_(0, 1)
    return input_param


def first_caption(caption):
    return caption.split(",")[0]

def add_caption(image, caption, color="white", font_size=20):
    caption = first_caption(caption)
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

    mask = (mask > 0).float()  # 1 means umasked, 0 means dont need to modify

    # do style transfer first
    # Original was 1000 iterations
    x_s = style_transfer(x, style_refer, mask, content_w=1, style_w=4000, num_iters=100)
    # si(
    #     torch.cat([x, mask, style_refer, x_s * mask + x * (1 - mask)], -1),
    #     save_path + f"/{exp_name}_style_trans.png",
    # )

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
        f"Results for Attack Style\n \
            Original = {y_pred.item()} : {d_dict[y_pred.item()]}\n  \
            Adv Region = {x_adv_no_purify_pred} : {d_dict[x_adv_no_purify_pred]}\n  \
            Adv Purified Region = {y_final} : {d_dict[y_final]}"
    )

    save_name = save_path + f"/{first_caption(d_dict[y_pred.item()])}_to_{first_caption(d_dict[y_final])}.png"
    images_tensor =  torch.cat(
            [
                torch.cat(
                    [
                        add_caption(x, f"O: {d_dict[y_pred.item()]}"),
                        #    add_caption(style_refer, f"T: {d_dict[target_pred]}"),
                        add_caption(style_refer, f"T: {d_dict[target_pred]}"),
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
        )
    # print("SAVING TO", save_name)
    # si(
    #     images_tensor,
    #     save_name
    # )

    # Return whether or not the attack was succesful (label was changed)
    res_adv = 1 if d_dict[y_pred.item()] != d_dict[x_adv_no_purify_pred] else 0
    res_pur = 1 if d_dict[y_pred.item()] != d_dict[y_final] else 0
    return res_adv, res_pur, images_tensor


import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import to_pil_image

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

def show_all_images(dataset: Dataset):
    """Display all images from a PyTorch dataset."""
    fig = plt.figure(figsize=(10, 10))  # Adjust the figure size as needed

    # Loop through all the images in the dataset
    for i, (image, label) in enumerate(dataset):
        ax = fig.add_subplot(1, len(dataset), i + 1)  # Adjust subplot grid parameters as necessary
        ax.set_title(f'Label: {label}')
        ax.axis('off')  # Turn off axis numbers and ticks

        # Convert tensor image to PIL for displaying
        image = ToPILImage()(image)
        ax.imshow(image)

    plt.show()


def create_mask(image, device='cpu'):
    was_tensor = isinstance(image, torch.Tensor)

    # If it's a tensor, check if it has a batch dimension
    if was_tensor:
        if image.ndim == 4 and image.shape[0] == 1:  # Check for single image batch
            image = image.squeeze(0)  # Remove the batch dimension
        image = transforms.ToPILImage()(image)

    # Convert PIL Image to OpenCV format
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and detail in the image
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Thresholding to create a binary image
    _, thresholded = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours from the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask with the same dimensions as the input image, initialized to zero (black)
    mask = np.zeros_like(gray_image)
    
    # Fill all detected contours
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Convert mask back to PyTorch tensor
    mask_tensor = torch.tensor(mask, device=device).to(dtype=torch.float32)
    mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dimension

    return mask_tensor

def display_tensor(tensor):
    # Assuming tensor is a single-channel tensor
    if tensor.ndim > 2 and tensor.shape[0] == 1:  # Check for a single-channel image
        tensor = tensor.squeeze(0)  # Remove channel dimension if it's the only channel

    # Convert to PIL for easy display
    image = to_pil_image(tensor)

    # Use matplotlib to display the image
    plt.imshow(image, cmap='gray')  # Use gray scale color map for single channel data
    plt.axis('off')  # Hide axes
    plt.show()


full_dataset = get_dataset("imagenet", "test")
device = "mps"

def save_checkpoint(state, filename="checkpoint6.pth.tar"):
    torch.save(state, filename)

def load_checkpoint(filename="checkpoint6.pth.tar"):
    if os.path.isfile(filename):
        return torch.load(filename)
    else:
        return None

def run_random():
    checkpoint = load_checkpoint()
    if checkpoint:
        print("Loading from checkpoint...")
        start_index = checkpoint['next_index']
        num_success_adv = checkpoint['num_success_adv']
        num_success_pur = checkpoint['num_success_pur']
        percents_adv = checkpoint['percents_adv']
        percents_pur = checkpoint['percents_pur']
        num_success_adv_no_mask = checkpoint['num_success_adv_no_mask']
        num_success_pur_no_mask = checkpoint['num_success_pur_no_mask']
        percents_adv_no_mask = checkpoint['percents_adv_no_mask']
        percents_pur_no_mask = checkpoint['percents_pur_no_mask']
        num_success_adv_global = checkpoint['num_success_adv_global']
        num_success_pur_global = checkpoint['num_success_pur_global']
        percents_adv_global = checkpoint['percents_adv_global']
        percents_pur_global = checkpoint['percents_pur_global']
        print("CHECKPOINT IS", checkpoint)
    else:
        start_index = 0
        num_success_adv = num_success_pur = num_success_adv_no_mask = num_success_pur_no_mask = 0
        num_success_adv_global = num_success_pur_global = 0
        percents_adv = []
        percents_pur = []
        percents_adv_no_mask = []
        percents_pur_no_mask = []
        percents_adv_global = []
        percents_pur_global = []

    n = 10
    for index in tqdm(range(n)):
        start_index += 1
        i = random.randint(0, len(full_dataset))
        # Get a different index from i for the style
        j = random.randint(0, len(full_dataset))
        if j == i:
            j = (j + 1) % len(full_dataset)

        x, y = full_dataset[i]
        x = x[None,].to(device)
        y = torch.tensor(y)[None,].to(device)

        mask = create_mask(x, device=device)

        style_refer = full_dataset[j][0][None, ...].to(device)

        print("=================BEGIN ATTACK REGION STYLE================")
        results = Attack_Region_Style(
            f"rand_{index}", x, mask, style_refer, "resnet50", device, "ddim40", t=4, eps=16, iter=10
        )
        num_success_adv += results[0]
        num_success_pur += results[1]

        # print("LEN PERCENTS_ADV BEFORE:", len(percents_adv))
        # print("NUM SUCCESSES ADV", num_success_adv)
        # print("START INDEX:", start_index, "LENGTH", len(percents_adv))
        percents_adv.append(num_success_adv / start_index)
        # print("LEN PERCENTS_ADV AFTER:", len(percents_adv))
        # print("PERCENTS ADV", percents_adv)
        percents_pur.append(num_success_pur / start_index)

        print("=================BEGIN ATTACK REGION STYLE NO MASK================")
        no_mask = load_png(f"../data/advcam_dataset/other_imgs/seg/no-mask.jpg", 224)[
            None, ...
        ].to(device)

        results_no_mask = Attack_Region_Style(
            f"rand_{x}", x, no_mask, style_refer, "resnet50", device, "ddim40", t=4, eps=16, iter=10
        )

        num_success_adv_no_mask += results_no_mask[0]
        num_success_pur_no_mask += results_no_mask[1]

        percents_adv_no_mask.append(num_success_adv_no_mask / start_index)
        percents_pur_no_mask.append(num_success_pur_no_mask / start_index)

        print("=================BEGIN GLOBAL ATTACK================")
        results_global = Attack_Global(
            x, y, "resnet50", device, "ddim40", 4, eps=16, iter=10, name="attack_global", skip=200
        )
        num_success_adv_global += results_global[0]
        num_success_pur_global += results_global[1]

        percents_adv_global.append(num_success_adv_global / start_index)
        percents_pur_global.append(num_success_pur_global / start_index)

        save_images_path = f"vis/custom_style/test/comb_{time.time()}.png"
        print("SAVING COMBINED STACK TO", save_images_path)
        # print("RESULTS", type(results[2]), type(results_no_mask[2]), type(results_global[2]))
        # print("VALS", results[2].shape, results_no_mask[2].shape, results_global[2].shape)

        # Verifying conditions are met
        assert results[2].dtype == results_no_mask[2].dtype == results_global[2].dtype, "Data types do not match"
        assert results[2].device == results_no_mask[2].device == results_global[2].device, "Tensors are on different devices"

        # Concatenate tensors
        # print("RESULTS [2] IS", results[2].shape, results[2])
        # combined_images_tensor = torch.stack(
        #     [results[2], results_no_mask[2], results_global[2]],
        #     dim=0
        # )
        combined_images_tensor = torch.cat(
            [results[2], results_no_mask[2], results_global[2]],
            dim=1
        )
        si(combined_images_tensor, save_images_path)
        if (index + 1) % 1 == 0:
            save_checkpoint({
                'next_index': start_index,
                'num_success_adv': num_success_adv,
                'num_success_pur': num_success_pur,
                'percents_adv': percents_adv,
                'percents_pur': percents_pur,
                'num_success_adv_no_mask': num_success_adv_no_mask,
                'num_success_pur_no_mask': num_success_pur_no_mask,
                'percents_adv_no_mask': percents_adv_no_mask,
                'percents_pur_no_mask': percents_pur_no_mask,
                'num_success_adv_global': num_success_adv_global,
                'num_success_pur_global': num_success_pur_global,
                'percents_adv_global': percents_adv_global,
                'percents_pur_global': percents_pur_global,
            })
            print(f"Checkpoint saved at iteration {index + 1}")

    
    iters = list(range(1, start_index + 1))
    print("PERCENTS ADV", percents_adv)
    plt.plot(iters, percents_adv, label="Masked")
    plt.plot(iters, percents_pur, label="Masked Purified")
    plt.plot(iters, percents_adv_no_mask, label="No Mask")
    plt.plot(iters, percents_pur_no_mask, label="No Mask Purified")
    plt.plot(iters, percents_adv_global, label="Global")
    plt.plot(iters, percents_pur_global, label="Global Purified")
    plt.xlabel("Number of Images")
    plt.ylabel("Success Rate (%)")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    save_path = f'vis/custom_style/test/{time.time()}.png'
    mp(save_path)
    print("SAVING TO", save_path)
    plt.savefig(save_path)
    plt.show()
    plt.close()



def run_gun():
    target_label = 413 # Corresponds to "assault rifle"
    # target_label = 681
    target_indices = [i for i, label in enumerate(full_dataset.targets) if label == target_label]
    # Create a subset of the full dataset using the target indices
    filtered_dataset = Subset(full_dataset, target_indices)

    # Show the first image in the filtered dataset
    # show_all_images(filtered_dataset)

    for i in range(len(filtered_dataset)):
        print("RUNNING FOR GUN", i)
        x, y = filtered_dataset[i]
        x = x[None,].to(device)
        mask = create_mask(x, device=device)
        # mask = load_png(f"../data/advcam_dataset/other_imgs/seg/no-mask.jpg", 224)[
        #     None, ...
        # ].to(device)
        style_refer = load_png(
            f"../data/advcam_dataset/other_imgs/img/black_fish.jpg", 224
        )[None, ...].to(device)

        Attack_Region_Style(
            f"gun_{i}", x, mask, style_refer, "resnet50", device, "ddim40", t=4, eps=16, iter=10
        )

run_random()