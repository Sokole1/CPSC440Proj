from utils import load_png
from archs import get_archs, IMAGENET_MODEL
from pred import d as d_dict
import glob

# from load_dm import get_imagenet_dm_conf
# from dataset import get_dataset
# from utils import *
# import torch
# from torch.utils.data import Subset
# import torchvision
# from tqdm.auto import tqdm
# import random
# from archs import get_archs, IMAGENET_MODEL
# from advertorch.attacks import LinfPGDAttack
# import matplotlib.pylab as plt
# import time
# import glob
import torch
from image_captions import *
# from attack_tools import gen_pgd_confs


# def save_images():
#     print(
#         f"Results for image {exp_name}:\n \
#             Original = {y_pred.item()} : {d_dict[y_pred.item()]}\n  \
#             Adv Region = {x_adv_no_purify_pred} : {d_dict[x_adv_no_purify_pred]}\n  \
#             Adv Purified Region = {y_final} : {d_dict[y_final]}"
#     )

#     print("SAVING TO", save_path + f"/{exp_name}_final{y_final}.png")
#     si(
#         torch.cat(
#             [
#                 torch.cat(
#                     [
#                         add_caption(x, f"O: {d_dict[y_pred.item()]}"),
#                         #    add_caption(style_refer, f"T: {d_dict[target_pred]}"),
#                         add_caption(style_refer, "Target (T)"),
#                         add_caption(x_s, f"ST: {d_dict[x_s_pred]}"),
#                         add_caption(
#                             x_adv_diff_region,
#                             f"Diff-PGD: {d_dict[x_adv_no_purify_pred]}",
#                         ),
#                         add_caption(
#                             x_adv_diff_p_region, f"Purified: {d_dict[y_final]}"
#                         ),
#                         add_caption(mask, "mask"),

#                         for x in
#                     ],
#                     -1,
#                 )
#             ],
#             -2,
#         ),
#         save_path + f"/{exp_name}_final{y_final}.png",
#     )

def test_single_image(x, classifier, device, y, y_target=None, name='attack_physics'):
    classifier = get_archs(classifier, 'imagenet')

    classifier = classifier.to(device)
    classifier.eval()
    # load image from image_path to classify image
    # print(name)
    # print(classifier(x).argmax(1))
    # out = classifier(x)

    y_pred = classifier(x).argmax(1)

    # print(y_pred, classifier(x_adv).argmax(1), classifier(pred_x0).argmax(1))

    # print(f"Results for image {name}:\n \
    #             Original = {y} : {d_dict[y]}\n  \
    #             Predicted = {y_pred.item()} : {d_dict[y_pred.item()]}\n  \
    #             Target = {y_target} : {d_dict[y_target]}")

    if y_target is not None:
        if y_pred == y_target:
            return 2 if y_pred == y_target else 0, y_pred, d_dict[y_pred.item()]
        return 1 if y_pred == y_target else 0, y_pred, d_dict[y_pred.item()]
    return 1 if classifier(x).argmax(1) == y else 0, y_pred, d_dict[y_pred.item()]

def load_physical_samples(device, category = '620+530', name='panda on laptop'):
    bkg_dir = f'data/samples/physical/{category}/*/'
    # print("HERE")
    print(glob.glob(bkg_dir+'/*.png') + glob.glob(bkg_dir+'/*.jpg'))
    bkg_all = glob.glob(bkg_dir+'/*.jpg')
    bkg_all += glob.glob(bkg_dir+'/*.png')

    # each model needs a success rate score, and two for if there is a target

    original_labels = set()
    models = {}

    images = []
    for bkg_selected in bkg_all:
        y_target = None
        path = bkg_selected.split('/')
        if not ('original' in bkg_selected) and not ('none' in bkg_selected):
            continue
        iterations = path[-4]
        label = path[-3]
        arr = label.split('+')
        y = int(arr[0])
        original_labels.add(d_dict[y])
        if len(arr) > 1:
            y_target = int(arr[1])
        adv_model = path[-2]
        # print(path[-2])
        bkg = load_png(bkg_selected, 224)[None, ...].to(device)
        is_correct, y_pred, y_pred_label = test_single_image(bkg.clone(), 'resnet50', device, y=y, y_target=y_target, name=adv_model)
        original_labels.add(y_pred_label)

    for bkg_selected in bkg_all:
        path = bkg_selected.split('/')
        if 'original' in bkg_selected or 'none' in bkg_selected:
            continue

        label = path[-3]
        arr = label.split('+')
        y = int(arr[0])
        if len(arr) > 1:
            y_target = int(arr[1])
        adv_model = path[-2]
        print(path[-2])
        bkg = load_png(bkg_selected, 224)[None, ...].to(device)
        is_correct, y_pred, y_pred_label = test_single_image(bkg.clone(), 'resnet50', device, y=y, y_target=y_target, name=adv_model)

        if adv_model not in models:
            models[adv_model] = [0, 0, 0]
        if y_target is not None and y_pred == y_target:
            models[adv_model][2] += 1
        elif not (y_pred_label in original_labels):
            models[adv_model][1] += 1
        else:
            models[adv_model][0] += 1

        images.append(
            add_caption(
                bkg,
                f"{adv_model}: {d_dict[y_pred.item()]}",
            )
        )
    N = len(models.keys())
    x = models.keys()
    failed = np.zeros(N)
    change_success = np.zeros(N)
    target_success = np.zeros(N)

    # loop through models dict
    for i, key in enumerate(x):
        value = models[key]
        print(f"Model: {key}\n \
            Fail: {value[0]}\n \
            Targeted Success Rate: {value[2]}\n \
            Changed: {value[1]}")
        failed[i] = value[0]
        change_success[i] = value[1]
        target_success[i] = value[2]

    if y_target is not None:
        plt.bar(x, target_success, color='b')
        plt.bar(x, change_success, bottom = target_success, color='orange')
        plt.bar(x, failed, bottom = target_success + change_success, color='g')
        plt.legend(['Predicted target', 'Class Change', 'Attack Failed'])
    else:
        plt.bar(x, change_success, color='orange')
        plt.bar(x, failed, bottom = change_success, color='g')
        plt.legend(['Class Change', 'Attack Failed'])

    plt.xlabel("Adversarial Attack Methods")
    plt.ylabel("Image Count")
    plt.title(name)

    plt.savefig(f'./physical_{name.replace(" ", "_")}_success_rate.png')
    plt.close()
    si(images, f'./physical_samples_{name.replace(" ", "_")}.png')

    print(f"Original Labels: {original_labels}")

load_physical_samples('mps')
load_physical_samples('mps', category = '898', name='forest on water bottle')
bkg_dir = 'data/samples/physical/1400/'
print("HERE")
# print(glob.glob(bkg_dir+'/*/*.png') + glob.glob(bkg_dir+'/*/*.jpg'))
# def Test_Physics(image_path, classifier, device, y_pred, y_target=None, name='attack_physics'):
#     classifier = get_archs(classifier, 'imagenet')

#     classifier = classifier.to(device)
#     classifier.eval()
#     # load image from image_path to classify image
#     x = load_png(image_path, 224)[None, ].to(device)
#     # print(name)
#     print(classifier(x).argmax(1))
#     out = classifier(x)
#     loss = out[:, y_pred]

#     print(y_pred, classifier(x_adv).argmax(1), classifier(pred_x0).argmax(1))


#     if y_target is not None:
#         loss -= out[:, y_target]

#     loss.backward()