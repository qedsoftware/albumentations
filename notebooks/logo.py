import os

import cv2
import albumentations as A
from albumentations.augmentations import functional as F
import imageio


def main():
    image = cv2.imread('logo.png')

    keyframes = []

    keyframes.append(image)
    keyframes.append(F.rot90(image, 3))
    keyframes.append(A.HueSaturationValue(p=1)(image=keyframes[-1])['image'])
    keyframes.append(F.rot90(image, 2))
    keyframes.append(A.Blur(p=1, blur_limit=15)(image=keyframes[-1])['image'])
    keyframes.append(F.rot90(image, 1))
    # keyframes.append(image)

    # keyframes.append(image)
    # keyframes.append(A.HorizontalFlip(p=1)(image=keyframes[-1])['image'])
    # keyframes.append(image)
    # keyframes.append(A.VerticalFlip(p=1)(image=keyframes[-1])['image'])
    # keyframes.append(image)
    # keyframes.append(A.HorizontalFlip(p=1)(image=keyframes[-1])['image'])
    # keyframes.append(A.InvertImg(p=1)(image=keyframes[-1])['image'])
    # keyframes.append(A.VerticalFlip(p=1)(image=keyframes[-1])['image'])
    # keyframes.append(A.InvertImg(p=1)(image=keyframes[-1])['image'])

    sequence = []
    n_frames = 15

    for i in range(len(keyframes) - 1):
        from_image = keyframes[i]
        to_image = keyframes[i + 1]

        sequence.append(from_image)

        for op in range(n_frames):
            alpha = 1. - op / float(n_frames - 1)
            beta = op / float(n_frames - 1)
            frame = cv2.addWeighted(from_image, alpha, to_image, beta, 0)
            sequence.append(frame)

        sequence.append(to_image)

    # Make nice loop
    from_image = keyframes[-1]
    to_image = keyframes[0]

    sequence.append(from_image)

    for op in range(n_frames):
        alpha = 1 - op / float(n_frames - 1)
        beta = op / float(n_frames - 1)
        frame = cv2.addWeighted(from_image, alpha, to_image, beta, 0)
        sequence.append(frame)

    sequence.append(to_image)

    filenames = []

    os.makedirs('logo_frames', exist_ok=True)
    for i, frame in enumerate(sequence):
        fname = os.path.join('logo_frames', f'frame_{i:04d}.png')
        filenames.append(fname)
        cv2.imwrite(fname, frame)
        cv2.imshow("Logo", frame)
        cv2.waitKey(30)

    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('logo.gif', images)


if __name__ == '__main__':
    main()
