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

# from attack_tools import gen_pgd_confs

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

    print(f"Results for image {name}:\n \
                Original = {y} : {d_dict[y]}\n  \
                Predicted = {y_pred.item()} : {d_dict[y_pred.item()]}\n  \
                Target = {y_target} : {d_dict[y_target]}")

    if y_target is not None:
        return y_pred == y_target
    return classifier(x).argmax(1) == y

def load_physical_samples(device):
    bkg_dir = 'data/samples/physical/*/*/*/'
    print("HERE")
    print(glob.glob(bkg_dir+'/*.png') + glob.glob(bkg_dir+'/*.jpg'))
    bkg_all = glob.glob(bkg_dir+'/*.jpg')
    bkg_all += glob.glob(bkg_dir+'/*.png')

    for bkg_selected in bkg_all:
        path = bkg_selected.split('/')
        iterations = path[-4]
        label = path[-3]
        arr = label.split('+')
        y = int(arr[0])
        if len(arr) > 1:
            y_target = int(arr[1])
        adv_model = path[-2]
        print(path[-2])
        bkg = load_png(bkg_selected, 224)[None, ...].to(device)
        test_single_image(bkg.clone(), 'resnet50', device, y=y, y_target=y_target, name=adv_model)

load_physical_samples('mps')
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