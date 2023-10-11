import argparse


def init_args():

    parser = argparse.ArgumentParser(
        description='Generate explanations using I-GOS and iGOS++.'
    )

    parser.add_argument(
        '--model',
        metavar='M',
        type=str,
        choices=['vgg19', 'resnet50', 'm-rcnn', 'f-rcnn', 'yolov3spp'],
        default='resnet50',
        help='The model to use for making predictions.')

    parser.add_argument(
        '--data',
        metavar='D',
        type=str,
        required=True,
        help='the path to the data to be explained.')
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['imagenet', 'coco', 'voc'],
        default='imagenet',
        help='The dataset to use for making predictions.')

    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='Shuffle the dataset.')

    parser.add_argument(
        '--size',
        type=int,
        default=28,
        help='The resolution of mask to be generated.')

    parser.add_argument(
        '--input_size',
        type=int,
        default=224,
        help='The input size to the network.')

    parser.add_argument(
        '--num_samples',
        type=int,
        default=5000,
        help='The number of samples to run explanation on.')

    parser.add_argument(
        '--manual_seed',
        type=int,
        default=63,
        help='The manual seed for experiments.')

    parser.add_argument(
        '--method',
        required=True,
        type=str,
        choices=['I-GOS', 'iGOS+', 'iGOS++'],
        default='I-GOS'
    )

    parser.add_argument(
        '--opt',
        required=True,
        type=str,
        choices=['LS', 'NAG'],
        default='NAG',
        help='The optimization algorithm.'
    )

    parser.add_argument(
        '--diverse_k',
        type=int,
        default=2)

    parser.add_argument(
        '--init_val',
        type=float,
        default=0.,
        help='The initialization value.')

    parser.add_argument(
        '--init_posi',
        type=int,
        default=0,
        help='The initialization position.')

    parser.add_argument(
        '--L1',
        type=float,
        default=1
    )

    parser.add_argument(
        '--L2',
        type=float,
        default=20
    )

    parser.add_argument(
        '--ig_iter',
        type=int,
        default=20)

    parser.add_argument(
        '--iterations',
        type=int,
        default=15
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=1000
    )

    return parser.parse_args()
