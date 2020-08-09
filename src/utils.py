import cv2


#  https://github.com/albu/albumentations/blob/master/albumentations/augmentations/functional.py
def pad(img, min_height, min_width):
    height, width = img.shape[:2]

    if height < min_height:
        h_pad_top = int((min_height - height) / 2.0)
        h_pad_bottom = min_height - height - h_pad_top
    else:
        h_pad_top = 0
        h_pad_bottom = 0

    if width < min_width:
        w_pad_left = int((min_width - width) / 2.0)
        w_pad_right = min_width - width - w_pad_left
    else:
        w_pad_left = 0
        w_pad_right = 0

    img = cv2.copyMakeBorder(img, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right,
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))

    assert img.shape[0] == max(min_height, height)
    assert img.shape[1] == max(min_width, width)

    return img


#  https://github.com/albu/albumentations/blob/master/albumentations/augmentations/functional.py#L120
def center_crop(img, crop_height, crop_width):
    height, width = img.shape[:2]
    if height < crop_height or width < crop_width:
        raise ValueError(
            'Requested crop size ({crop_height}, {crop_width}) is '
            'larger than the image size ({height}, {width})'.format(
                crop_height=crop_height,
                crop_width=crop_width,
                height=height,
                width=width,
            )
        )
    y1 = (height - crop_height) // 2
    y2 = y1 + crop_height
    x1 = (width - crop_width) // 2
    x2 = x1 + crop_width
    img = img[y1:y2, x1:x2]
    return img
