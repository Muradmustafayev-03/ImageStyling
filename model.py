from image_processing import InputImage, OutputImage
from tensorflow.python.keras import models
import tensorflow as tf
import numpy as np

NUM_STYLE_LAYERS = 5
CONTENT_LAYERS = ['block5_conv2']
STYLE_LAYERS = [f'block{i + 1}_conv1' for i in range(NUM_STYLE_LAYERS)]


class Model:
    def __init__(self):
        self.model = self.get_model()
        for layer in self.model.layers:
            layer.trainable = False
        self.optimizer = tf.optimizers.Adam(learning_rate=5, epsilon=1e-1)

    @staticmethod
    def get_model():
        """ Creates our model with access to intermediate layers.

        This function will load the VGG19 model and access the intermediate layers.
        These layers will then be used to create a new model that will take input image
        and return the outputs from these intermediate layers from the VGG model.

        Returns:
          returns a keras model that takes image inputs and outputs the style and
            content intermediate layers.
        """
        # Load our model. We load pretrained VGG, trained on imagenet data
        vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        # Get output layers corresponding to style and content layers
        style_outputs = [vgg.get_layer(name).output for name in STYLE_LAYERS]
        content_outputs = [vgg.get_layer(name).output for name in CONTENT_LAYERS]
        model_outputs = style_outputs + content_outputs
        # Build model
        return models.Model(vgg.input, model_outputs)

    @staticmethod
    def get_content_loss(base_content, target):
        return tf.reduce_mean(tf.square(base_content - target))

    @staticmethod
    def gram_matrix(input_tensor):
        # We make the image channels first
        channels = int(input_tensor.shape[-1])
        a = tf.reshape(input_tensor, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a=True)
        return gram / tf.cast(n, tf.float32)

    def get_style_loss(self, base_style, gram_target):
        """Expects two images of dimension h, w, c"""
        gram_style = self.gram_matrix(base_style)
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def get_feature_representations(self, content_path, style_paths: list):
        """Helper function to compute our content and style feature representations.

        This function will simply load and preprocess the content image from its path
        and a list of style images from their paths. Then it will feed them through the
        network to obtain the outputs of the intermediate layers.

        Arguments:
          content_path: The path to the content image.
          style_paths: A list of paths to the style images.

        Returns:
          returns a list of style features and the content features.
        """
        # Load our content image in
        content_image = InputImage(content_path).preprocessed

        # Load and preprocess style images
        style_images = [InputImage(style_path).preprocessed for style_path in style_paths]

        # batch compute content and style features
        style_outputs = [self.model(style_image) for style_image in style_images]
        content_outputs = self.model(content_image)

        # Get the style and content feature representations from our model
        content_features_list = []
        for style_output in style_outputs:
            content_features_list.append([content_layer[0] for content_layer in style_output[NUM_STYLE_LAYERS:]])

        style_features_list = []
        for style_output in style_outputs:
            style_features_list.append([style_layer[0] for style_layer in style_output[:NUM_STYLE_LAYERS]])

        content_features = [content_layer[0] for content_layer in content_outputs[NUM_STYLE_LAYERS:]]

        return content_features, content_features_list, style_features_list

    def compute_loss(self, loss_weights, init_image, gram_style_features_list, content_features_list, content_features):
        """This function will compute the total loss, style loss, content loss, and total variational loss.

        Arguments:
          loss_weights: Loss weights
          init_image: Our initial base image. This image is what we are updating with
            our optimization process. We apply the gradients wrt the loss we are
            calculating to this image.
          gram_style_features_list: List of precomputed gram matrices corresponding to the
            defined style layers of interest in each style image.
          content_features: Precomputed outputs from defined content layers of interest.

        Returns:
          returns the total loss, style loss, content loss, and total variational loss
        """
        style_weight, content_weight = loss_weights
        model_outputs = self.model(init_image)

        style_output_features = model_outputs[:NUM_STYLE_LAYERS]
        content_output_features = model_outputs[NUM_STYLE_LAYERS:]

        style_score = 0
        content_score = 0

        # Accumulate style losses from all layers and all style images
        # Here, we equally weight each contribution of each loss layer and each style image
        for i in range(len(gram_style_features_list)):
            gram_style_features = gram_style_features_list[i]
            content_features_for_image = content_features_list[i]
            style_loss_per_image = 0
            content_loss_per_image = 0
            for target_style, comb_style in zip(gram_style_features, style_output_features):
                style_loss_per_image += self.get_style_loss(comb_style[0], target_style) / float(NUM_STYLE_LAYERS)
            for target_content, comb_content in zip(content_features_for_image, content_output_features):
                content_loss_per_image += self.get_content_loss(comb_content[0], target_content)
            style_score += style_loss_per_image * content_loss_per_image / len(gram_style_features_list)

        # Accumulate content losses from all layers
        for target_content, comb_content in zip(content_features, content_output_features):
            content_score += self.get_content_loss(comb_content[0], target_content)

        style_score *= style_weight
        content_score *= content_weight

        # Get total loss
        loss = style_score + content_score
        return loss, style_score, content_score

    def compute_grads(self, cfg):
        with tf.GradientTape() as tape:
            all_loss = self.compute_loss(**cfg)
        # Compute gradients wrt input image
        total_loss = all_loss[0]
        return tape.gradient(total_loss, cfg['init_image']), all_loss

    def run_style_transfer(self, content_path, style_paths, num_iterations=1000, content_weight=1, style_weight=1e-6):
        # Get the style and content feature representations (from our specified intermediate layers)
        content_features, content_features_list, style_features_list = \
            self.get_feature_representations(content_path, style_paths)
        gram_style_features = [[self.gram_matrix(style_feature) for style_feature in img_style_features]
                               for img_style_features in style_features_list]

        # Set initial image
        init_image = InputImage(content_path).preprocessed
        init_image = tf.Variable(init_image, dtype=tf.float32)

        # Store our best result
        best_loss, best_img = float('inf'), None

        # Create a nice config
        loss_weights = (style_weight, content_weight)
        cfg = {
            'loss_weights': loss_weights,
            'init_image': init_image,
            'gram_style_features_list': gram_style_features,
            'content_features_list': content_features_list,
            'content_features': content_features
        }

        norm_means = np.array([103.939, 116.779, 123.68])
        min_vals = -norm_means
        max_vals = 255 - norm_means

        for i in range(num_iterations):
            grads, all_loss = self.compute_grads(cfg)
            loss, style_score, content_score = all_loss

            self.optimizer.apply_gradients([(grads, init_image)])
            clipped = tf.clip_by_value(init_image, min_vals, max_vals)
            init_image.assign(clipped)

            if loss < best_loss:
                best_loss = loss
                best_img = OutputImage(init_image.numpy()).img
                print(f'Iteration: {i+1}')
                print(f'Total loss: {loss:.4e}, '
                      f'style loss: {style_score:.4e}, '
                      f'content loss: {content_score:.4e}')

        return best_img, best_loss
