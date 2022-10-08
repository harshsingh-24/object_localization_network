# How to run ? python extras/folder_testing.py data/custom_images

import os
from argparse import ArgumentParser
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

def main():
    parser = ArgumentParser()
    parser.add_argument('directory', help='Image Folder')
    args = parser.parse_args()
    
    config = 'configs/oln_box/oln_box.py'
    checkpoint = 'trained_weights/latest.pth'
    model = init_detector(config, checkpoint, device='cuda:0')

    count = 100 
    for filename in os.listdir(args.directory):

        if filename.lower().endswith(('.jpg')) and count != 0:
            file_location = args.directory + '/' + filename
            result = inference_detector(model, file_location)

            # output file location: custom building
            splittedImageLoc = file_location.split("/")
            temparg = 'output/o' + splittedImageLoc[len(splittedImageLoc) - 1]; 

            model.show_result(
            file_location,
            result,
            score_thr=0.7,
            show=True,
            wait_time=0,
            fig_size=(15, 10),
            win_name='result',
            bbox_color=(72, 101, 241),
            text_color=(72, 101, 241),
            out_file=temparg)

            count = count - 1

if __name__ == '__main__':
    main()