"""
main file to call the explanations methods and run experiments, given a pre-trained
model and a data loader.
© copyright Tyler Lawson, Saeed khorram. https://github.com/saeed-khorram/IGOS
"""

import torchvision.models as models
from torch.autograd import Variable

from args import init_args
from utils import *
from methods_helper import *
from methods import IGOS, iGOS_p, iGOS_pp


def gen_explanations(model, dataloader, args):

    model.eval()

    out_dir = init_logger(args)

    if args.method == "I-GOS":
        method = IGOS
    elif args.method == "iGOS+":
        method = iGOS_p
    elif args.method == "iGOS++":
        method = iGOS_pp
    else:
        raise ValueError("the method does not exist. Choose from IGOS or iGOS++")

    eprint(f'Size is {args.size}x{args.size}')

    i = 0
    total_del, total_ins, total_time = 0, 0, 0

    for data in dataloader:

        # unpack images and turn them into variables
        images, blurs = data
        images, blurs = Variable(images).cuda(), Variable(blurs).cuda()

        _, labels = torch.max(model(images), 1)

        now = time.time()

        # generate masks
        masks = method(
            model,
            images=images.detach(),
            baselines=blurs.detach(),
            labels=labels,
            size=args.size,
            iterations=args.ig_iter,
            ig_iter=args.iterations,
            L1=args.L1,
            L2=args.L2,
            alpha=args.alpha,
        )

        total_time += time.time() - now

        # Calculate the scores for the masks
        del_scores, ins_scores, del_curve, ins_curve, index = metric(
            images,
            blurs,
            masks.detach(),
            model,
            labels,
            step=max(1, args.size ** 2 // 50),
            size=args.size
        )

        # save heatmaps, images, and del/ins curves
        save_heatmaps(masks, images, args.size, i, out_dir)
        save_curves(del_curve, ins_curve, index, i, out_dir)
        save_images(images, i, out_dir, classes, labels)

        # log info
        total_del += del_scores.sum().item()
        total_ins += ins_scores.sum().item()
        i += images.shape[0]

        eprint(
            f'{args.method:6} ({i} samples)'
            f' Deletion (Avg.): {total_del / i:.05f}'
            f' Insertion (Avg.): {total_ins / i:.05f}'
            f' Time (Avg.): {total_time / i:.03f}'
        )

        if i >= args.num_samples:
            break

    model.train()


if __name__ == "__main__":

    args = init_args()
    eprint(f"args:\n {args}")

    torch.manual_seed(args.manual_seed)

    init(args.input_size)
    init_sns()

    classes = get_imagenet_classes()

    dataset = ImageSet(args.data, blur=True)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=4
    )

    eprint("Loading the model...")

    if args.model == 'vgg19':
        model = models.vgg19(pretrained=True, progress=True).cuda()

    elif args.model == 'resnet50':
        model = models.resnet50(pretrained=True, progress=True).cuda()

    else:
        raise ValueError("Model not defined.")

    for child in model.parameters():
        child.requires_grad = False

    eprint(f"Model({args.model}) successfully loaded!\n")

    gen_explanations(model, data_loader, args)

