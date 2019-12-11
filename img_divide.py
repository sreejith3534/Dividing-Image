import numpy as np
from PIL import Image
from resizeimage import resizeimage


def resize_img(img_path):
    """
    parameter:
    img_path : path to image
    returns:
    img : resized img_array

    """
    _img = Image.open(img_path)
    img_resized = resizeimage.resize_cover(_img, [6000, 6000], validate=False)
    img = np.array(img_resized)
    return img


def divide_img(im_arr, out_path, shape):
    """
    parameter:
    im_arr : arr of image file.
    out_path = path to save the divided image.
    shape = dimension to which image should be divided. For eg. shape = 10 will divide an image into 10*10 = 100 images.
    returns:
    img : an array with position from which it has been cut with it's array values.
    """
    all_array_pos = []
    img_count = 0
    y = 0
    height = shape  # size for feature mapping while using pre-trained model.
    x_pos = 0
    try:
        while y <= im_arr.shape[1]:
            x = 0
            width = shape  # size for feature mapping while using pre-trained model.
            all_img = 0
            try:
                while x <= im_arr.shape[0]:
                    _im = im_arr[x:width, y:height]
                    im = Image.fromarray(_im)
                    labeled = [str(x_pos) + str(all_img), im]
                    all_array_pos.append(labeled)
                    im.save(out_path + str(img_count) + '_out.jpg', 'JPEG')  # save the image.
                    img_count += 1
                    all_img += 1
                    x += shape
                    width += shape
            except ValueError:
                print('end of iteration')
            y += shape
            height += shape
            x_pos += 1
    except SystemError:
        print('out of bound, file saved')
    return all_array_pos


if __name__ == '__main__':
    all_array = divide_img(img, '../_test/')
    print(len(all_array))
    print(all_array[0])
    print(all_array[19])
    print(all_array[20])
    print(all_array[21])
