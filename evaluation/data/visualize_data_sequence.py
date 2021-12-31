import os

import click
import cv2
import numpy as np
from PIL import Image


def color_pixel(img, y, x):
    try:
        img[y][x] = [255, 0, 0]
    except IndexError:
        pass


def color_around(img, x, y):
    for i in range(-1, 2):
        for j in range(-1, 2):
            color_pixel(img, y + i, x + j)


def create_video_from_sequence(sequence_dir: str):
    video_file_path = os.path.join(sequence_dir, "sequence_visualized.mp4")
    images_dir = os.path.join(sequence_dir, "observations")

    data = np.load(os.path.join(sequence_dir, "data.npz"))
    actions = data["actions"]
    rewards = data["rewards"]

    images_list = os.listdir(images_dir)
    images_list.sort()
    images_list = images_list[:-1]

    assert len(images_list) > 0
    assert len(images_list) == actions.shape[0] and len(images_list) == rewards.shape[0]

    img = np.asarray(Image.open(os.path.join(images_dir, images_list[0])))

    size = (img.shape[0], img.shape[1])
    fps = 25

    video_out = cv2.VideoWriter(video_file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[0], size[1]), isColor=True)

    for i, img_file_path in enumerate(images_list):
        x, y = actions[i]
        img = np.asarray(Image.open(os.path.join(images_dir, img_file_path)))

        color_around(img, x, y)

        for _ in range(fps):
            video_out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    video_out.release()


@click.command()
@click.option("-s", "--single", "dir_mode", flag_value="single", default=True)
@click.option("-m", "--multiple", "dir_mode", flag_value="multiple")
@click.option("-d", "--directory", type=str, required=True)
def main(dir_mode, directory: str):
    if dir_mode == "single":
        print(f"Creating video in single directory '{directory}'")
        create_video_from_sequence(directory)
    else:
        print(f"Creating multiple videos for sequences in main directory '{directory}'")
        for sub_dir in os.listdir(directory):
            print(f"Starting video creation for sub directory '{sub_dir}'")
            create_video_from_sequence(os.path.join(directory, sub_dir))
            print(f"Finished sub directory '{sub_dir}'")


if __name__ == '__main__':
    main()
