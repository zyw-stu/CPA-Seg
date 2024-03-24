# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmengine.model import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model, show_result_pyplot

def main():
    parser = ArgumentParser()
    parser.add_argument('--img',
                        default='H:/ACDC/cityscapes_acdc_all/leftImg8bit/val/snow/GOPR0606_frame_000213_rgb_anon.png',
                        help='Image file')
    parser.add_argument('--config',default='../configs/deeplabv3plus/acdc_deeplabv3plus_config.py', help='Config file') # deeplabv3plus
    parser.add_argument('--checkpoint', default='./path/to/pretrained_models/best.pth',help='Checkpoint file') # deeplabv3plus
    #parser.add_argument('--config', default='../configs/segformer/acdc_segformer_config.py', help='Config file')
    #parser.add_argument('--checkpoint', default='./path/to/segformer/best.pth', help='Checkpoint file')

    parser.add_argument('--out-file', default=None,help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=1,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--with-labels',
        action='store_true',
        default=False,
        help='Whether to display the class labels.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    # test a single image
    result = inference_model(model, args.img)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result,
        title=args.title,
        opacity=args.opacity,
        with_labels=args.with_labels,
        draw_gt=False,
        show=False if args.out_file is not None else True,
        out_file=args.out_file)


if __name__ == '__main__':
    main()
