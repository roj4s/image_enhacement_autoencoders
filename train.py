import os
from keras.models import Model
from keras.layers import Input, UpSampling2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import tensorflow as tf
import random
import numpy as np
from matplotlib import pyplot as plt
from autoencoder import Autoencoder_x2_s
from autoencoder import Autoencoder_x2
import argparse
import datetime
import keras

models = {
    'Autoencoder_x2_s': Autoencoder_x2_s,
    'Autoencoder_x2': Autoencoder_x2
}

model_names = ",".join(list(models.keys()))

seed = 33
random.seed = seed
np.random.seed = seed
tf.random.set_seed(seed)


gpus = tf.config.experimental.list_physical_devices('GPU')
print(f"Found {len(gpus)} gpus")
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    i = 0
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      tf.config.experimental.set_virtual_device_configuration(
                  gpu,
                  [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)]
      )
      print(f"\tEnabling memory growth for gpu: {i}")
      i -= -1
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)



def configure_for_performance(ds, buffer_size=32, batch_size=32):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=buffer_size)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=buffer_size)
  return ds

def get_xy_paths(x_root, y_root, scale='x2'):
  y_paths = []
  x_paths = []
  for yi_name in os.listdir(y_root):
    yi_path = os.path.join(y_root, yi_name)
    xi_name = yi_name.replace('x1', scale).split('.png')[0] + scale + '.png'
    xi_path = os.path.join(x_root, xi_name)
    if os.path.exists(xi_path):
      y_paths.append(yi_path)
      x_paths.append(xi_path)

  return x_paths, y_paths

def decode_img(path):
  img = tf.io.read_file(path)
  return tf.io.decode_png(img, channels=3)

def process_pair_paths(paths):
  return decode_img(paths[0]), decode_img(paths[1])

def psnr(x, y):
    return tf.image.psnr(x, y, 500)

def ssim(x, y):
    return tf.image.ssim(x, y, 500)

def ssim_multiscale(x, y):
    return tf.image.ssim_multiscale(x, y, 500)


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description='Train flow for cnn '\
                                     'degradation model autoencoder')
    parser.add_argument('x_root', type=str, help="Directory containing input "\
                        "images")
    parser.add_argument("y_root", type=str, help="Directory containing golden "\
                        "images")
    parser.add_argument('--scale', type=str, help="Scale", default='x2')
    parser.add_argument('--model', type=str, help="Model name", default='Autoencoder_x2_s')
    parser.add_argument('--epochs', type=int, help="Epochs to train model, "\
                        "default is 200",
                        default=200)
    parser.add_argument('--checkpoints-output', help='Optional, directory to '\
                        'output checkpoints', type=str,
                        default=f'/tmp/checkpoints_{timestamp}')
    parser.add_argument('--tensorboard-output', help='Optional, directory to '\
                        'output tensorboard logs', type=str,
                        default=f'/tmp/tensorboard_{timestamp}')
    parser.add_argument('--show-dataset', action='store_true',
                    help='Plot dataset samples', default=False)

    args = parser.parse_args()
    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False


    if not args.model in models:
        print(f"Model name {args.model} must be one of: {model_names}")
        return 1

    print(f"Seting up training for model: {args.model}")
    print(f"X Root: {args.x_root}")
    print(f"Y Root: {args.y_root}")

    x_paths, y_paths = get_xy_paths(args.x_root, args.y_root)
    pairs_paths = [[x_path, y_path] for x_path, y_path in zip(x_paths, y_paths)]
    pair_count = len(x_paths)

    print(f"Found {pair_count} pairs")

    ds = tf.data.Dataset.from_tensor_slices(pairs_paths)
    ds = ds.shuffle(pair_count, reshuffle_each_iteration=False)

    val_size = int(pair_count * 0.2)
    train_ds = ds.skip(val_size)
    val_ds = ds.take(val_size)

    train_count = tf.data.experimental.cardinality(train_ds).numpy()
    val_count = tf.data.experimental.cardinality(val_ds).numpy()
    print(f"\nTrain count: {train_count}, Validation count: {val_count}")

    train_ds = train_ds.map(process_pair_paths)
    val_ds = val_ds.map(process_pair_paths)

    train_ds = configure_for_performance(train_ds, batch_size=10)
    val_ds = configure_for_performance(val_ds, batch_size=10)

    for x_i, y_i in train_ds.take(1):
      print("Xi shape: ", x_i.numpy().shape)
      print("Yi shape: ", y_i.numpy().shape)

    if args.show_dataset:
        print("Samples:")
        for f in train_ds.take(5):
          print(f)
          print(f.numpy().shape)

        x_batch, y_batch = next(iter(train_ds))
        plt.figure(figsize=(40, 16))

        for i in range(5):
            j = i * 2 + 1
            k = j + 1
            plt.subplot(5, 2, j)
            plt.imshow(x_batch[i])

            plt.subplot(5, 2, k)
            plt.imshow(y_batch[i])

        plt.show()

    model = models[args.model]()
    model.compile(optimizer='adam', loss='mean_squared_error',
                  metrics=[psnr, ssim, ssim_multiscale])
    model.summary()

    print(f"Tensorboard outputs to: {args.tensorboard_output}")
    tensorboard_callback  = keras.callbacks.TensorBoard(log_dir=args.tensorboard_output,
                                                          histogram_freq=1)
    file_writer_cm = tf.summary.create_file_writer(args.tensorboard_output
                                                   + '/output_samples')
    def show_output(epoch, logs):
        x_batch, y_batch = next(iter(val_ds))
        _input = np.array([x_batch[0]])
        print(f"Input shape: {_input.shape}")
        img = model.predict(_input)
        with file_writer_cm.as_default():
                tf.summary.image(f"output_{epoch}", img, step=epoch)

    show_output_callback = keras.callbacks.LambdaCallback(on_epoch_end=show_output)

    print(f"Chekpoints outputs to: {args.checkpoints_output}")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.checkpoints_output,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
              callbacks=[model_checkpoint_callback, tensorboard_callback,
                         show_output_callback])

if __name__ == "__main__":
    main()
