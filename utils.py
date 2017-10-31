import os
import glob
import scipy.misc
import scipy.ndimage
import numpy as np

def get_image_paths(dir_path):
    """
    Get image paths from dir
    """
    dir_path = os.path.join(os.getcwd(), dir_path)
    image_paths = glob.glob(os.path.join(dir_path, '*.bmp'))
    return image_paths

def imread(image_path, is_grayscale=True):
    """
    Read image from path
    """
    if is_grayscale:
        return scipy.misc.imread(image_path, flatten=True, mode='YCbCr').astype(np.float)
    else:
        return scipy.misc.imread(image_path, mode='YCbCr').astype(np.float)

def mod_crop(image, scale=3):
    """
    modulo crop the image with the scale
    """
    h, w = image.shape[0], image.shape[1]
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    return image[:h, :w]

def process_image(image, scale=3):
    """
    Normalize
    Aplly bicubic interpolation
    
    Args:
        file_path: path to single image
    Return:
        - inp: image applied bicubic interpolation (low resolution)
        - label: image with original resolution (high resolution)
    """
    label = mod_crop(image, scale)
    label = label/255 #normalize

    inp = scipy.ndimage.interpolation.zoom(label, (1./scale), prefilter=False)
    inp = scipy.ndimage.interpolation.zoom(inp, (scale/1.), prefilter=False)

    return inp, label

def slicing_image(inp, label, I, L, stride):
    """
    Slice inp image (low resolution) to obtain multiple sub-image of size (kernel x kernel)
    with the step being stride
    
    Args:
        - inp: input image (low resolution)
        - label: image of high resolution
        - I: filter size of to slice inp
        - L: filter size to slice label
        - stride: step size to move filter
    Return
        - sub_inputs: sub-images from inp
        - sub_labels: sub_images from label
    """
    sub_inputs = []
    sub_labels = []
    h, w = inp.shape[0], inp.shape[1]
    offset = abs(I - L)//2 #offset to identify where to slice label
    
    for hh in range(0, h-I+1, stride):
        for ww in range(0, w-I+1, stride):
            sub_input = inp[hh:hh+I, ww:ww+I]
            sub_label = label[hh+offset:hh+offset+L, ww+offset:ww+offset+L]

            # Make channel value
            sub_input = sub_input.reshape(I, I, 1)
            sub_label = sub_label.reshape(L, L, 1)

            sub_inputs.append(sub_input)
            sub_labels.append(sub_label)
    return sub_inputs, sub_labels

def make_input(config):
    """
    Produce image and label from config

    Args:
        Config: config to produce image and label

    Return:
        - inputs: data for the model
        - labels: groud true for the model
    """
    inputs = []
    labels = []

    dir_path = config['dir_path']
    scale = config['scale']
    is_grayscale = config['is_gray']
    I = config['input_size']
    L = config['label_size']
    stride = config['stride']

    image_paths = get_image_paths(dir_path)
    
    for path in image_paths:
        image = imread(path, is_grayscale)
        inp, label = process_image(image, scale)
        sub_inputs, sub_labels = slicing_image(inp, label, I, L, stride)
        inputs += sub_inputs
        labels += sub_labels

    inputs = np.asarray(inputs) # shape (N, I, I, 1)
    labels = np.asarray(labels) # shape (N, L, L, 1)
    return inputs, labels

# testing
if __name__ == "__main__":
    paths = get_image_paths('Train')
    image = imread(paths[0])
    print(image)
    inp, label = process_image(image)
    scipy.misc.imsave('inp.png', inp)
    scipy.misc.imsave('label.png', label)
    print(inp)
    print(label)
