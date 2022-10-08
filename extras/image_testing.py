# Program for visualizing output on a single image with bounding boxes

from argparse import ArgumentParser
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.7, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results

    # output file location: custom building
    splittedImageLoc = args.img.split("/")
    temparg = 'output/o' + splittedImageLoc[len(splittedImageLoc) - 1]; 

    model.show_result(
        args.img,
        result,
        score_thr=0.7,
        show=True,
        wait_time=0,
        fig_size=(15, 10),
        win_name='result',
        bbox_color=(72, 101, 241),
        text_color=(72, 101, 241),
        out_file=temparg)


if __name__ == '__main__':
    main()