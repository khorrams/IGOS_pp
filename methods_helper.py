"""
Helper function for the IGOS explanation methods.
© copyright Tyler Lawson, Saeed khorram. https://github.com/saeed-khorram/IGOS
"""

import torch


# Initializes the upsampling function for the upscale function
def init(out_size):
    """
    Initializes output size for the upsampler.
    :param out_size:
    :return:
    """
    upscale.out_size = out_size
    upscale.up = torch.nn.UpsamplingBilinear2d(size=(out_size, out_size)).cuda()


def tv_norm(image, beta=2):
    """
    Calculates the total variation.
    :param image:
    :param beta:
    :return:
    """
    image = image[:, 0, :, :]
    a = torch.mean(torch.abs((image[:, :-1, :] - image[:, 1:, :]).view(image.shape[0], -1)).pow(beta), dim=1)
    b = torch.mean(torch.abs((image[:, :, :-1] - image[:, :, 1:]).view(image.shape[0], -1)).pow(beta), dim=1)
    return a + b


def bilateral_tv_norm(image, mask, tv_beta=2, sigma=1):
    """
        Calculates the bilateral total variation.

    :param image:
    :param mask:
    :param tv_beta:
    :param sigma:
    :return:
    """
    # tv term
    mask_ = mask[:, 0, :]
    a = torch.mean(torch.abs((mask_[:, :-1, :] - mask_[:, 1:, :]).view(mask.shape[0], -1)).pow(tv_beta), dim=1)
    b = torch.mean(torch.abs((mask_[:, :, :-1] - mask_[:, :, 1:]).view(mask.shape[0], -1)).pow(tv_beta), dim=1)
    # bilateral tv in the image space
    up_mask_ = upscale(mask)

    bil_a = torch.mean(torch.exp(-(image[:, :, :-1, :] - image[:, :, 1:, :]).mean(dim=1) ** 2 / sigma).view(mask.shape[0], -1)
                       * torch.abs((up_mask_[:, :, :-1, :] - up_mask_[:, :, 1:, :]).view(up_mask_.shape[0], -1)).pow(tv_beta), dim=1)
    bil_b = torch.mean(torch.exp(-(image[:, :, :, :-1] - image[:, :, :, 1:]) ** 2 / sigma).mean(dim=1).view(mask.shape[0], -1)
                       * torch.abs((up_mask_[:, :, :, :-1] - up_mask_[:, :, :, 1:]).view(up_mask_.shape[0], -1)).pow(tv_beta), dim=1)
    return 0.5 * (a + b + bil_a + bil_b)


def upscale(masks):
    """
        Upsamples masks and expands it to the same dimensions as the image

    values are set using the init() function
    :param masks:
    :return:
    """
    return upscale.up(masks).expand((-1,1,upscale.out_size,upscale.out_size))  # TODO


def interval_score(model, model_name, image, baseline, label, up_masks, num_iter, output_func, noise=True):
    """
        Computes the score of masked image in a straight line
        path from baseline to masked image, with num_iter intervals.

    :param model:
    :param model_name:
    :param image:
    :param baseline:
    :param label:
    :param up_masks:
    :param num_iter:
    :param noise:
    :param output_func:
    :return:
    """
    # The intervals to approximate the integral over
    intervals = torch.linspace(1/num_iter, 1, num_iter, requires_grad=False).cuda().view(-1, 1, 1, 1)
    interval_masks = up_masks.unsqueeze(1) * intervals
    local_images = phi(image.unsqueeze(1), baseline.unsqueeze(1), interval_masks)

    if noise:
        local_images = local_images + torch.randn_like(local_images) * .2

    # Shape of image tensor when viewed in batch form
    new_shape = torch.tensor(image.shape) * torch.tensor(intervals.shape)

    if model_name in ['m-rcnn', 'f-rcnn']: 
        losses = torch.cat([out['scores'] for out in model(local_images.view(*new_shape))]).view(-1, num_iter, 1)

    elif model_name == 'yolov3spp':
        losses = model(local_images.view(*new_shape)).view(-1, num_iter, 1)

    else:
        losses = output_func(local_images.view(*new_shape), model).view(image.shape[0], num_iter, -1)
        losses = torch.gather(losses, 2, label.cuda().view(-1, 1).expand(-1, num_iter).view(-1, num_iter, 1))

    return losses / num_iter


def integrated_gradient(model, model_name, image, baseline, label, up_masks, num_iter, output_func=None, noise=True):
    """
        Calculates and backprops the integrated gradient.
        Does not have the original mask, so does not return the gradient

    :param model:
    :param model_name:
    :param image:
    :param baseline:
    :param label:
    :param up_masks:
    :param num_iter:
    :param noise:
    :param output_func:
    :return:
    """
    for i in range(image.shape[0]):
        loss = interval_score(
                model,
                model_name, 
                image[i].unsqueeze(0),
                baseline[i].unsqueeze(0),
                label[i].unsqueeze(0),
                up_masks[i].unsqueeze(0),
                num_iter,
                output_func,
                noise,
                )
        loss.sum().backward(retain_graph=True)


def line_search(masks, total_grads, loss_func, alpha=8, beta=0.0001, decay=0.2,):
    """
        Computes a line search in batch. Works by starting far in the direction of total_grads and works
        backward until all meet the target condition or their corresponding alpha value is below some value.
        Uses loss_func for the target condition.

    :param masks:
    :param total_grads:
    :param loss_func:
    :param alpha:
    :param beta:
    :param decay:
    :return:
    """
    # Speed up computations, reduce memory usage, and ensure no autograd
    # graphs are created
    with torch.no_grad():
        i = 0
        mod = len(masks.shape) - 3
        num_inputs = masks.shape[0]
        # The indices of masks that still need their alphas updated
        indices = torch.ones(num_inputs, dtype=torch.bool).cuda()
        # Create initial alpha values for each mask
        alphas = torch.ones(num_inputs).cuda() * alpha

        up_masks = upscale(masks.view(-1,*masks.shape[mod:])).view(-1, *masks.shape[1:mod], 1, upscale.out_size, upscale.out_size)

        # Compute the base loss used in the condition
        base_losses = loss_func(up_masks, masks, indices).view(-1)
        t = -beta * (total_grads ** 2).view(num_inputs, -1).sum(dim=1).view(num_inputs)

        while True:
            # Create a new mask with the updated alpha value to
            # see if it meets condition
            new_masks = torch.clamp(masks[indices] - alphas[indices].view(-1,*(1,) * mod,1,1) * total_grads[indices], 0, 1)
            up_masks = upscale(new_masks.view(-1,*masks.shape[mod:])).view(-1,*masks.shape[1:mod], 1, upscale.out_size, upscale.out_size)
            # Calculate new losses
            losses = loss_func(up_masks, new_masks, indices).view(-1)
            # Get indices for each alpha that meets the condition for
            # their corresponding mask
            indices[indices] = losses > base_losses[indices] + alphas[indices] * t[indices]
            # Same for this, but for if the alpha values are too low (\alpha_l)
            indices[indices] *= (alphas[indices] >= 0.00001)
            # Break out of the loop if all alpha values satisfy the condition
            # or are too low
            if not indices.sum():
                break
            # Otherwise update alphas
            alphas[indices] *= decay
            i += 1
    return alphas.view(-1,1,1,1)


def phi(img, baseline, mask):
    """
        Composes an image from img and baseline according to the mask values.

    :param img:
    :param baseline:
    :param mask:
    :return:
    """
    return img.mul(mask) + baseline.mul(1-mask)


def softmax_output(inputs, model):
    """
        Applies softamx over the output of the model.

    :param inputs:
    :param model:
    :return:
    """
    return torch.nn.Softmax(dim=1)(model(inputs))


def logit_output(inputs, model):
    """
        Simply returns the output of the model, given an input.

    :param inputs:
    :param model:
    :return:
    """
    return model(inputs)


def metric(image, baseline, mask, model, model_name, label, label_i, pred_data, size=28,):
    """
        Calculates the deletion/insertion scores/curves given the image and generated masks.

    :param image:
    :param baseline:
    :param mask:
    :param model:
    :param model_name:
    :param label:
    :param label_i:
    :param pred_data:
    :param size:
    :return:
    """
    with torch.no_grad():
        # The dimensions for the image
        img_size = image.shape[-1]
        # Compute the total number of pixels in a mask
        mask_pixels = torch.prod(torch.tensor(mask.shape[1:])).item()
        if model_name == 'm-rcnn':
            num_pixels = max(1, int(pred_data['masks'][label_i].sum() * ((size / img_size) ** 2))) * 3
            if num_pixels > mask_pixels:
                num_pixels = mask_pixels
        elif model_name in ['f-rcnn', 'yolov3spp']:
            x1 = pred_data['boxes'][label_i][0] * (size / img_size)
            y1 = pred_data['boxes'][label_i][1] * (size / img_size)
            x2 = pred_data['boxes'][label_i][2] * (size / img_size)
            y2 = pred_data['boxes'][label_i][3] * (size / img_size)
            num_pixels = max(1, int((x2 - x1) * (y2 - y1))) * 3
            if num_pixels > mask_pixels:
                num_pixels = mask_pixels
        else:
            num_pixels = torch.prod(torch.tensor(mask.shape[1:])).item()
        # Compute the step size
        step=max(1, num_pixels // 50)
        # Used for indexing with batch sizes
        l = torch.arange(image.shape[0])
        # The unmasked score
        og_scores = score_output(image, model, model_name, l, label)
        # The baseline score
        blur_scores = score_output(baseline, model, model_name, l, label)
        # Initial values for the curves
        del_curve = [og_scores]
        ins_curve = [blur_scores]
        index = [0.]

        # True_mask is used to hold 1 or 0. Either show that pixel or blur it.
        true_mask = torch.ones((mask.shape[0], mask_pixels)).cuda()
        del_scores = torch.zeros(mask.shape[0]).cuda()
        ins_scores = torch.zeros(mask.shape[0]).cuda()
        # Sort each mask by values and store the indices.
        elements = torch.argsort(mask.view(mask.shape[0], -1), dim=1)

        for pixels in range(0, num_pixels, step):

            # Get the indices used in this iteration
            indices = elements[l,pixels:pixels+step].squeeze().view(image.shape[0], -1)
            # Set those indices to 0
            true_mask[l, indices.permute(1,0)] = 0
            up_mask = upscale(true_mask.view(-1, 1, size,size))
            # Mask the image for deletion
            del_image = phi(image, baseline, up_mask)
            # Calculate new scores
            outputs = score_output(del_image, model, model_name, l, label)
            del_curve.append(outputs)
            index.append((pixels+step)/num_pixels)
            del_scores += outputs * step if pixels + step < num_pixels else\
                num_pixels - pixels

            # Mask the image for insertion
            ins_image = phi(baseline, image, up_mask)

            # Calculate the new scores
            outputs = score_output(ins_image, model, model_name, l, label)

            ins_curve.append(outputs)
            ins_scores += outputs * step if pixels + step < num_pixels else\
                num_pixels - pixels

        # Force scores between 0 and 1.
        del_scores /= num_pixels
        ins_scores /= num_pixels

        del_curve = list(map(lambda x: [y.item() for y in x], zip(*del_curve)))
        ins_curve = list(map(lambda x: [y.item() for y in x], zip(*ins_curve)))

    return del_scores, ins_scores, del_curve, ins_curve, index


def score_output(image, model, model_name, l, label):
    """
        Get score of this image.

    :param image:
    :param model:
    :param model_name:
    :param l:
    :param label:
    :return:
    """
    if model_name in ['m-rcnn', 'f-rcnn']:
        return (model(image))[0]['scores']
    elif model_name == 'yolov3spp':
        return model(image)[0][0]
    else:
        return torch.nn.Softmax(dim=1)(model(image))[l,label]