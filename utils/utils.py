from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def deprocess(img):
    img = img.add_(1).div_(2)

    return img


def process_mask(img):
    for s in range(0, img.size[0]):
        for t in range(0, img.size[1]):
            pixel = img.getpixel((s, t))
            if pixel[0] >= 128 and pixel[1] >= 128 and pixel[2] >= 128:
                img.putpixel((s, t), (255, 255, 255))
            else:
                img.putpixel((s, t), (0, 0, 0))

    return img
