# from utils import load_png
# from archs import get_archs, IMAGENET_MODEL

# def test_single_image(x, classifier, device, y_pred, y_target=None, name='attack_physics'):
#     classifier = get_archs(classifier, 'imagenet')

#     classifier = classifier.to(device)
#     classifier.eval()
#     # load image from image_path to classify image
#     print(name)
#     print(classifier(x).argmax(1))
#     out = classifier(x)
#     loss = out[:, y_pred]

#     print(y_pred, classifier(x_adv).argmax(1), classifier(pred_x0).argmax(1))


#     if y_target is not None:
#         loss -= out[:, y_target]

#     loss.backward()
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