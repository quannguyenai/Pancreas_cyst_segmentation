import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt):

    pred[pred > 0] = 1
    gt[gt > 0] = 1

    if pred.sum() == 0 and gt.sum() == 0:
        return 1, 0

    if pred.sum() == 0 or gt.sum() == 0:
        return 0, 0

    dice = metric.binary.dc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)

    return dice, hd95


def test_single_volume(image, label, model, classes, patch_size=[256,256]):

    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()

    prediction = np.zeros_like(label)

    model.eval()

    for ind in range(image.shape[2]):

        slice = image[:, :, ind]

        fg_mask = slice > 0

        if np.sum(fg_mask) > 0:
            mean = slice[fg_mask].mean()
            std = slice[fg_mask].std()
            slice = (slice - mean) / (std + 1e-8)

        x, y = slice.shape

        slice_resized = zoom(
            slice,
            (patch_size[0] / x, patch_size[1] / y),
            order=1
        )

        input = torch.from_numpy(slice_resized)\
                    .unsqueeze(0)\
                    .unsqueeze(0)\
                    .float()\
                    .cuda()

        with torch.no_grad():

            output = model(input)

            if isinstance(output, (list,tuple)):
                output = output[0]

            out = torch.argmax(
                torch.softmax(output, dim=1),
                dim=1
            ).squeeze(0)

            out = out.cpu().numpy()

        pred = zoom(
            out,
            (x / patch_size[0], y / patch_size[1]),
            order=0
        )

        prediction[:, :, ind] = pred

    metric_list = []

    for i in range(1, classes):
        metric_list.append(
            calculate_metric_percase(
                prediction == i,
                label == i
            )
        )

    return metric_list