import os
# Use CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import tensorflow_addons as tfa
from IPython.display import clear_output
import matplotlib.pyplot as plt
import pathlib
import time
import math

data_dir = "C:/Users/tj13/PyCharm_Datasets/Oxford-IIIT-sub/"
result_dir = "C:/Users/tj13/Results/unet_5"

#
# Load the raw data
#
data_dir = pathlib.Path(data_dir)
print("data_dir = ", data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print("image_count = ", image_count)

mask_count = len(list(data_dir.glob('*/*.png')))
print("mask_count = ", mask_count)

def get_file_list(file_type):
    list_file = tf.data.Dataset.list_files(file_type, shuffle=False)
    list_file = list_file.shuffle(len(list_file), reshuffle_each_iteration=False)
    print(file_type, " = ", len(list_file))
    for f in list_file.take(5):
        print(f.numpy())
    return list_file

# A tensor of strings
list_image = get_file_list(str(data_dir/'images/*.jpg'))
print("list_image = ", list_image)

#
# Build the data set
#
split_size = int(image_count * 0.2)
train_ds = list_image.skip(split_size)
test_ds = list_image.take(split_size)

def get_samples(ds, rate=10.0):
    return ds.filter(lambda x: tf.random.uniform(()) < rate)

train_ds_flipping = get_samples(train_ds)
train_ds_rotation = get_samples(train_ds)
train_ds_brightness = get_samples(train_ds)
train_ds_contrast = get_samples(train_ds)
train_ds_flipping_v = get_samples(train_ds)
train_ds_rotation_270 = get_samples(train_ds)
train_ds_crop = get_samples(train_ds)
train_ds_rotation_45 = get_samples(train_ds)
train_ds_rotation_45N = get_samples(train_ds)
# train_ds_erasing = get_samples(train_ds)
train_ds_hue = get_samples(train_ds)
train_ds_saturation = get_samples(train_ds)

IMAGE_HEIGHT = IMAGE_WIDTH = 128

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask

def process_path(file_path):
    # load the raw data from the file as a string then convert to jpg
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])

    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The last is the file name
    file_name = tf.strings.split(parts[-1], ".")

    # load mask from the file as a string
    file_path = tf.constant(str(data_dir/"annotations")) + tf.constant("/") + file_name[0] + tf.constant(".png")
    msk = tf.io.read_file(file_path)
    msk = tf.image.decode_png(msk, channels=1)
    msk = tf.image.resize(msk, [IMAGE_HEIGHT, IMAGE_WIDTH])
    return img, msk

def load_image(file_path):
    input_image, input_mask = process_path(file_path)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

def image_flipping(file_path):
    input_image, input_mask = process_path(file_path)
    input_image = tf.image.flip_left_right(input_image)
    input_mask = tf.image.flip_left_right(input_mask)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

def image_rotation(file_path):
    input_image, input_mask = process_path(file_path)
    input_image = tf.image.rot90(input_image)
    input_mask = tf.image.rot90(input_mask)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

def image_brightness(file_path):
    input_image, input_mask = process_path(file_path)
    input_image, input_mask = normalize(input_image, input_mask)
    input_image = tf.image.random_brightness(input_image, 0.2)
    input_image = tf.clip_by_value(input_image, 0., 1.)
    return input_image, input_mask

def image_contrast(file_path):
    input_image, input_mask = process_path(file_path)
    input_image, input_mask = normalize(input_image, input_mask)
    input_image = tf.image.random_contrast(input_image, 0.2, 2.0)
    input_image = tf.clip_by_value(input_image, 0., 1.)
    return input_image, input_mask

def image_flipping_v(file_path):
    input_image, input_mask = process_path(file_path)
    input_image = tf.image.flip_up_down(input_image)
    input_mask = tf.image.flip_up_down(input_mask)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

def image_rotation_270(file_path):
    input_image, input_mask = process_path(file_path)
    input_image = tf.image.rot90(input_image, k=3)
    input_mask = tf.image.rot90(input_mask, k=3)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

def random_crop(image1, image2):
    # Convert image2 to a shape (,,3)
    image2 = tf.concat([image2, image2, image2], axis=2)
    stacked_image = tf.stack([image1, image2], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, IMAGE_HEIGHT, IMAGE_WIDTH, 3])

    # Split the second cropped image to three tensor with a shape (,,1)
    image2 = tf.split(cropped_image[1], 3, axis=2)
    return cropped_image[0], image2[0]

def image_crop(file_path):
    input_image, input_mask = process_path(file_path)
    input_image, input_mask = random_crop(input_image, input_mask)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

def image_rotation_45(file_path):
    input_image, input_mask = process_path(file_path)
    input_image = tfa.image.rotate(input_image, math.pi/4.0, fill_mode='reflect')
    input_mask = tfa.image.rotate(input_mask, math.pi/4.0, fill_mode='reflect')
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

def image_rotation_45N(file_path):
    input_image, input_mask = process_path(file_path)
    input_image = tfa.image.rotate(input_image, -math.pi/4.0, fill_mode='reflect')
    input_mask = tfa.image.rotate(input_mask, -math.pi/4.0, fill_mode='reflect')
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

def image_hue(file_path):
    input_image, input_mask = process_path(file_path)
    input_image, input_mask = normalize(input_image, input_mask)
    input_image = tf.image.random_hue(input_image, 0.4)
    input_image = tf.clip_by_value(input_image, 0., 1.)
    return input_image, input_mask

def image_saturation(file_path):
    input_image, input_mask = process_path(file_path)
    input_image, input_mask = normalize(input_image, input_mask)
    input_image = tf.image.random_saturation(input_image, 5, 10)
    input_image = tf.clip_by_value(input_image, 0., 1.)
    return input_image, input_mask

# def image_erasing(file_path, erase_height=50, erase_width=50):
#     input_image, input_mask = process_path(file_path)
#
#     input_image = tf.expand_dims(input_image, axis=0)
#     input_image = tfa.image.random_cutout(input_image, (erase_height,erase_width), constant_values=0)
#     input_image = tf.squeeze(input_image)
#
#     input_mask = tf.expand_dims(input_mask, axis=0)
#     input_mask = tfa.image.random_cutout(input_mask, (erase_height,erase_width), constant_values=0)
#     input_mask = tf.squeeze(input_mask)
#     input_mask = tf.expand_dims(input_mask, axis=2) #adding channel axis
#
#     input_image, input_mask = normalize(input_image, input_mask)
#     return input_image, input_mask

AUTOTUNE = tf.data.AUTOTUNE

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(load_image, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(load_image, num_parallel_calls=AUTOTUNE)

train_ds_flipping = train_ds_flipping.map(image_flipping, num_parallel_calls=AUTOTUNE)
train_ds_rotation = train_ds_rotation.map(image_rotation, num_parallel_calls=AUTOTUNE)
train_ds_brightness = train_ds_brightness.map(image_brightness, num_parallel_calls=AUTOTUNE)
train_ds_contrast = train_ds_contrast.map(image_contrast, num_parallel_calls=AUTOTUNE)
train_ds_flipping_v = train_ds_flipping_v.map(image_flipping_v, num_parallel_calls=AUTOTUNE)
train_ds_rotation_270 = train_ds_rotation_270.map(image_rotation_270, num_parallel_calls=AUTOTUNE)
train_ds_crop = train_ds_crop.map(image_crop, num_parallel_calls=AUTOTUNE)
train_ds_rotation_45 = train_ds_rotation_45.map(image_rotation_45, num_parallel_calls=AUTOTUNE)
train_ds_rotation_45N = train_ds_rotation_45N.map(image_rotation_45N, num_parallel_calls=AUTOTUNE)
# train_ds_erasing = train_ds_erasing.map(image_erasing, num_parallel_calls=AUTOTUNE)
train_ds_hue = train_ds_hue.map(image_hue, num_parallel_calls=AUTOTUNE)
train_ds_saturation = train_ds_saturation.map(image_saturation, num_parallel_calls=AUTOTUNE)


print(train_ds)

# Combine the original dataset with the augmented datasets

train_ds = train_ds.concatenate(train_ds_flipping)
train_ds = train_ds.concatenate(train_ds_rotation)
train_ds = train_ds.concatenate(train_ds_brightness)
train_ds = train_ds.concatenate(train_ds_contrast)
train_ds = train_ds.concatenate(train_ds_flipping_v)
train_ds = train_ds.concatenate(train_ds_rotation_270)
train_ds = train_ds.concatenate(train_ds_crop)
train_ds = train_ds.concatenate(train_ds_rotation_45)
train_ds = train_ds.concatenate(train_ds_rotation_45N)
# train_ds = train_ds.concatenate(train_ds_erasing)
train_ds = train_ds.concatenate(train_ds_hue)
train_ds = train_ds.concatenate(train_ds_saturation)

def get_size(ds):
    num_elements = 0
    for element in ds:
        num_elements += 1
    return num_elements

print("training size = ", get_size(train_ds))

def display(display_list, figure_title=None):
    if figure_title:
        fig = plt.figure(figsize=(15, 15), num=figure_title)
    else:
        fig = plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        
    plt.axis('off')
    # plt.show()
    fig.savefig(result_dir+figure_title+".png", bbox_inches='tight')
    plt.close()

# Show the first training sample's shape
sample_image, sample_mask = next(iter(train_ds.take(1)))
print("Image shape: ", sample_image.numpy().shape)
print("Mask shape: ", sample_mask.numpy().shape)
display([sample_image, sample_mask], "Sample_1")

BATCH_SIZE = 64
BUFFER_SIZE = 1000

# Build for performance
train_dataset = train_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_ds.batch(BATCH_SIZE)

#
# Define the model
#
def downsample(filters, kernel_size):
    result = tf.keras.Sequential()
    initializer = tf.random_normal_initializer(0., 0.02)
    # result.add(tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation='relu',
    #                                   kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.Conv2D(filters, kernel_size, padding='same'))#, use_bias=False))
    # result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.ReLU())
    result.add(tf.keras.layers.MaxPool2D())
    return result

def upsample(filters, kernel_size, apply_dropout=False):
    result = tf.keras.Sequential()
    # initializer = tf.random_normal_initializer(0., 0.02)
    # result.add(tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=2, padding='same',
    #                                            kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=2, padding='same'))#, use_bias=False))
    # result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())
    return result

def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[IMAGE_HEIGHT, IMAGE_WIDTH, 3])

    # Create the layers for the contraction part
    down_stack = [
        downsample(64, 3),  # 128x128 -> 64x64
        downsample(128, 3),  # 64x64 -> 32x32
        downsample(256, 3),  # 32x32 -> 16x16
        downsample(512, 3),  # 16x16 -> 8x8
        downsample(1024, 3),  # 8x8 -> 4x4
    ]

    # Connect the layers
    down_outputs = [down_stack[0](inputs)]
    for i in range(1, len(down_stack)):
        down_outputs.append(down_stack[i](down_outputs[i-1]))
        # print(x)

    x = down_outputs[-1] # 4x4
    reversed_outputs = reversed(down_outputs[:-1]) # 8x8, 16x16, 32x32, 64x64

    # Create the layers for the expansive part
    up_stack = [
        upsample(512, 3),  # 4x4 -> 8x8
        upsample(256, 3),  # 8x8 -> 16x16
        upsample(128, 3),  # 16x16 -> 32x32
        upsample(64, 3),  # 32x32 -> 64x64
    ]

    # Upsampling and establishing the skip connections
    for up, output in zip(up_stack, reversed_outputs):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, output])

    # This is the last layer of the model, 64x64 -> 128x128
    # last = tf.keras.layers.Conv2DTranspose(output_channels, 3, strides=2, padding='same',
    #                                        kernel_initializer=tf.random_normal_initializer(0., 0.02))
    last = tf.keras.layers.Conv2DTranspose(output_channels, 3, strides=2, padding='same')
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

#
# Train the model
#
OUTPUT_CHANNELS = 3

model = unet_model (OUTPUT_CHANNELS)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()
# tf.keras.utils.plot_model(model, show_shapes=True)

def create_mask(pred_mask):
    # pred_mask is in shape(1, 128, 128, 3) for three possible classes
    pred_mask = tf.argmax(pred_mask, axis=-1)

    # convert shape(1, 128, 128) to shape(1, 128, 128, 1) then return the first
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))], "Epoch_0")

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))],
                "Epoch_"+str(epoch+1))
        print('\nSample Prediction after epoch {}\n'.format(epoch+1))

EPOCHS = 20
# TRAIN_LENGTH = len(train_ds)
# STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
# TEST_LENGTH = len(test_ds)
# VAL_SUBSPLITS = 5
# VALIDATION_STEPS = TEST_LENGTH//BATCH_SIZE//VAL_SUBSPLITS

start_time = time.perf_counter()
model_history = model.fit(train_dataset,
                          epochs=EPOCHS,
                          validation_data=test_dataset,
                          callbacks=[DisplayCallback(), tf.keras.callbacks.TensorBoard(log_dir=result_dir)])
elapse_time = time.perf_counter() - start_time
print("Training took {:.2f} minutes.".format(elapse_time/60.))

# Show training results accuracy and loss
acc = model_history.history['accuracy']
val_acc = model_history.history['val_accuracy']

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

fig = plt.figure(figsize=(16, 8), num="Results")
plt.subplot(1, 2, 1)
plt.plot(model_history.epoch, acc, 'r', label='Training Accuracy')
plt.plot(model_history.epoch, val_acc, 'bo', label='Validation Accuracy')
plt.ylim([0.7, 1.0])
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(model_history.epoch, loss, 'r', label='Training Loss')
plt.plot(model_history.epoch, val_loss, 'bo', label='Validation Loss')
plt.ylim([0, 1])
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
# plt.show()
fig.savefig(result_dir+"Results.png", bbox_inches='tight')
plt.close()

#
# Make predictions
#
i = 1
for image, mask in test_dataset.take(10):
    pred_mask = model.predict(image)
    display([image[0], mask[0], create_mask(pred_mask)], "Prediction_" + str(i))
    i += 1
