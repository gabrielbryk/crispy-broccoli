import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19

base_image_path = "crispy-broccoli\\data\\0.jpg"
style_path = "crispy-broccoli\\data\\afremov\\afremov\\"
result_prefix = "cnn_61stack"

# Dimensions of the generated picture.
w, h = keras.preprocessing.image.load_img(base_image_path).size
rows = 250
cols = int(w*rows/h)

def preprocess_image(image_path):
    # Util function to open, resize and format pictures into appropriate tensors
    image = keras.preprocessing.image.load_img(
        image_path, target_size=(rows, cols)
    )
    image = keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = vgg19.preprocess_input(image)
    return tf.convert_to_tensor(image)


def deprocess_image(x):
    x = x.reshape((rows, cols, 3))
    x[:,:,0] += 103.939
    x[:,:,1] += 116.779
    x[:,:,2] += 123.68
    x = x[:,:,::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x

def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = rows*cols
    return tf.reduce_sum(tf.square(S - C))/(4.0*(channels**2)*(size**2))

def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))

def total_variation_loss(x):
    a = tf.square(x[:, :rows-1, :cols-1, :] - x[:, 1:, :cols-1, :])
    b = tf.square(x[:, :rows-1, :cols-1, :] - x[:, :rows-1, 1:, :])
    return tf.reduce_sum(tf.pow(a+b, 1.25))

model = vgg19.VGG19(weights="imagenet", include_top=False)
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

style_layer_names = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]
content_layer_name = "block5_conv2"


def compute_loss(output_image, base_image, style_image):
    input = tf.concat([base_image, style_image, output_image], axis=0)
    features = feature_extractor(input)
    loss = tf.zeros(shape=())

    #content loss
    layer_features = features[content_layer_name]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + (2.5e-8)* content_loss(base_image_features, combination_features)
    #style loss
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += ((1e-6)/len(style_layer_names)) * sl

    #total variation loss
    loss += (1e-6)* total_variation_loss(output_image)
    return loss

@tf.function
def compute_loss_and_grads(output_image, base_image, style_image):
    with tf.GradientTape() as tape:
        loss = compute_loss(output_image, base_image, style_image)
    grads = tape.gradient(loss, output_image)
    return loss, grads

optimizer = keras.optimizers.SGD(
    keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=100.0, decay_steps=200, decay_rate=0.96
    )
)

base_image = preprocess_image(base_image_path)
d = []
for i in range(1):
    d.append(preprocess_image(style_path+str(i)+".jpg"))
# d.append(preprocess_image(style_path))
style_image = tf.reduce_mean(d, axis=0)
# img = deprocess_image(style_image.numpy())
# fname = "style5cc_at_iteration_%d.png" % 0
# keras.preprocessing.image.save_img(fname, img)
# style_image = d[0]
output_image = tf.Variable(preprocess_image(base_image_path))

iterations = 4000
#training single style image
# for i in range(1, iterations + 1):
#     loss, grads = compute_loss_and_grads(output_image, base_image, style_image)
#     optimizer.apply_gradients([(grads, output_image)])
#     if i % 100 == 0:
#         print("Iteration %d: loss=%.2f" % (i, loss))
#         img = deprocess_image(output_image.numpy())
#         fname = "crispy-broccoli\\data\\test_result\\"+ result_prefix + "1_at_iteration_%d.png" % i
#         keras.preprocessing.image.save_img(fname, img)

# display(Image(result_prefix + "_at_iteration_4000.png"))

#training multiple style image
for j in range(61):
    optimizer = keras.optimizers.SGD(
    keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=100.0, decay_steps=200, decay_rate=0.96
        )
    )

    style_image = preprocess_image(style_path+str(j)+".jpg")
    for i in range(1, iterations + 1):
        loss, grads = compute_loss_and_grads(output_image, base_image, style_image)
        optimizer.apply_gradients([(grads, output_image)])
        if i % 100 == 0:
            print("Iteration %d: loss=%.2f" % (i, loss))
    img = deprocess_image(output_image.numpy())
    fname = "crispy-broccoli\\data\\test_result\\"+ result_prefix + str(j)+"_at_iteration_%d.png" % i
    keras.preprocessing.image.save_img(fname, img)
