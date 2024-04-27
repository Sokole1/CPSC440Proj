from utils import load_png
from archs import get_archs, IMAGENET_MODEL
import glob
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

    if y_target is not None:
        return y_pred == y_target
    return classifier(x).argmax(1) == y

def load_physical_samples(device):
    bkg_dir = 'data/samples/physical/'
    bkg_all = glob.glob(bkg_dir+'/*.jpg')
    bkg_all += glob.glob(bkg_dir+'/*.png')

    for bkg_selected in bkg_all:
        bkg = load_png(bkg_selected, 224)[None, ...].to(device)

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