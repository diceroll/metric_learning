import sys

import numpy as np
from PIL import Image
from scipy.misc import imresize
from scipy.ndimage import rotate


def mirror(img, label=None):
    img = img[:, ::-1, :]
    if label is not None:
        label = label[:, ::-1, :]

    return img, label


def flip(img, label=None):
    img = img[::-1, :, :]
    if label is not None:
        label = label[::-1, :, :]

    return img, label


def gamma_correction(img, label=None):
    rnd = np.random.rand() + 1
    if np.random.rand() < 0.5:
        rnd = 1 / rnd

    img = np.frompyfunc(lambda x, y: 255. * (x / 255.) ** y if x != 0 else x, 2, 1)(img, rnd)
    img = np.uint8(img)

    return np.float32(img), label


def random_rotate(img, label=None, cval=0, angle=(-5, 5)):
    '''
    angle: tuple or int or float
    '''

    if isinstance(angle, tuple):
        angle = np.random.rand() * (max(angle) - min(angle)) + min(angle)

    if angle != 0:
        img = rotate(img, angle, reshape=False)
        if label is not None:
            label = rotate(label, angle, reshape=False, order=0, cval=cval)

    return img, label


def random_resize(img, label=None, scale=(0.8, 1.2)):
    '''
    scale: tuple or int or float
    '''
    img_out = np.zeros_like(img)
    label_out = np.zeros_like(label)

    if isinstance(scale, tuple):
        if scale[0] <= 0 or scale[1] <= 0:
            raise ValueError
        scale = np.random.rand() * (max(scale) - min(scale)) + min(scale)

    img_resize = imresize(img, size=scale, interp='bicubic')
    h1, w1, _ = img.shape
    h2, w2, _ = img_resize.shape

    if scale > 1:
        sl_h = slice(h2 // 2 - h1 // 2, h2 // 2 - h1 // 2 + h1)
        sl_w = slice(w2 // 2 - w1 // 2, w2 // 2 - w1 // 2 + w1)
        img_out = img_resize[sl_h, sl_w]
    elif scale < 1:
        sl_h = slice(h1 // 2 - h2 // 2, h1 // 2 - h2 // 2 + h2)
        sl_w = slice(w1 // 2 - w2 // 2, w1 // 2 - w2 // 2 + w2)
        img_out[sl_h, sl_w] = img_resize

    if label is not None:
        label_resize = imresize(label, size=scale, interp='nearest')
        if scale > 1:
            label_out = label_resize[sl_h, sl_w]
        elif scale < 1:
            label_out[sl_h, sl_w] = label_resize

    return np.float32(img_out), np.int32(label_out)


def cutout(img, label=None, mask_size=None):
    img_out = img.copy()
    h, w, _ = img.shape

    if mask_size is None:
        mask_size = np.min((h, w)) // 3

    rnd1 = np.random.randint(-mask_size // 2, h - mask_size // 2)
    rnd2 = np.random.randint(-mask_size // 2, w - mask_size // 2)
    sl_h = slice(np.max((rnd1, 0)), np.min((rnd1 + mask_size, h)))
    sl_w = slice(np.max((rnd2, 0)), np.min((rnd2 + mask_size, w)))
    img_out[sl_h, sl_w] = img.mean()

    return img_out, label


def random_erasing(img, label=None, area_ratio=(0.02, 0.4), aspect_ratio=(0.3, 3)):
    img_out = img.copy()
    h, w, _ = img.shape
    area = h * w
    aspect = np.min(aspect_ratio) + np.random.rand() * np.abs(np.diff(aspect_ratio))

    min_size = np.max((np.sqrt(area * np.min(area_ratio) / aspect), 1 // aspect + (1 % aspect > 0)))
    max_size = np.min((np.sqrt(area * np.max(area_ratio) / aspect), w // aspect))
    mask_size = np.random.randint(int(min_size), int(max_size + 1))

    rnd1 = np.random.randint(0, h - mask_size)
    rnd2 = np.random.randint(0, w - int(mask_size * aspect))
    sl1 = slice(rnd1, rnd1 + mask_size)
    sl2 = slice(rnd2, rnd2 + int(mask_size * aspect))
    img_out[sl1, sl2] = np.random.randint(0, 256)

    return img_out, label


if __name__ == '__main__':
    '''
    Usage: python augmentation.py <Image Path>
    '''
    args = sys.argv
    img_path = args[1]
    img = {}
    img['org'] = np.array(Image.open(img_path), np.float32)

    img['mirror'], _ = mirror(img['org'])
    img['flip'], _ = flip(img['org'])
    img['gamma'], _ = gamma_correction(img['org'])
    img['rotate'], _ = random_rotate(img['org'], angle=90)
    img['resize'], _ = random_resize(img['org'])
    img['cutout'], _ = cutout(img['org'])
    img['random_erasing'], _ = random_erasing(img['org'])

    for k, v in img.items():
        print(k, v.shape)

    imgs = np.vstack((np.hstack((img['org'], img['mirror'], img['flip'], img['gamma'])),
                      np.hstack((img['rotate'], img['resize'], img['cutout'], img['random_erasing']))))
    print(imgs.shape)

    imgs = Image.fromarray(np.uint8(imgs))
    imgs.save('{}_augmentation.png'.format(img_path.split('.')[0]))
