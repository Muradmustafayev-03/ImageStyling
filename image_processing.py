from keras.utils import img_to_array
import tensorflow as tf
from PIL import Image
import numpy as np


class ImageProcessor:
    def __init__(self, img: np.array, preprocessed: np.array):
        self.img = img
        self.preprocessed = preprocessed

    def __add__(self, other):
        if not isinstance(other, ImageProcessor):
            raise TypeError('You can only add another ImageProcessor to an ImageProcessor')
        return ImageProcessor(np.add(self.img, other.img), np.add(self.preprocessed, other.preprocessed))

    def __str__(self):
        return str(self.img)

    def __repr__(self):
        return 'ImageProcessor object\n' \
               'Pixels matrix in RGB:\n' \
               f'{str(self)}\n' \
               'Pixels matrix in VGG19 format\n' \
               f'{str(self.preprocessed)}'


class InputImage(ImageProcessor):
    def __init__(self, path: str):
        img = self.load_img(path)
        super().__init__(img, self.preprocess(img))

    @staticmethod
    def load_img(path_to_img: str):
        print(f'Loading {path_to_img}')
        max_dim = 1024
        img = Image.open(path_to_img)
        img = img.resize((max_dim, max_dim), Image.ANTIALIAS)

        img = img_to_array(img)

        # We need to broadcast the image array such that it has a batch dimension
        img = np.expand_dims(img, axis=0)
        return img

    @staticmethod
    def preprocess(img: np.array):
        return tf.keras.applications.vgg19.preprocess_input(img)


class OutputImage(ImageProcessor):
    def __init__(self, preprocessed: np.array):
        super().__init__(self.restore_img(preprocessed), preprocessed)

    @staticmethod
    def restore_img(preprocessed_img):
        x = preprocessed_img.copy()
        if len(x.shape) == 4:
            x = np.squeeze(x, 0)
        assert len(x.shape) == 3, ("Input to restore image must be an image of "
                                   "dimension [1, height, width, channel] or [height, width, channel]")

        # perform the inverse of the preprocessing step
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        x = x[:, :, ::-1]

        x = np.clip(x, 0, 255).astype('uint8')
        return x
