import marimo

__generated_with = "0.11.2"
app = marimo.App(auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    return (os,)


@app.cell
def _():
    import tensorflow
    return (tensorflow,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""- List available devices""")
    return


@app.cell
def _(tensorflow):
    tensorflow.config.experimental.list_physical_devices()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""- Check if build with cuda (cuda compatible)""")
    return


@app.cell
def _(tensorflow):
    tensorflow.test.is_built_with_cuda()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 1. Import –  necessary modules and the dataset.""")
    return


@app.cell
def _():
    import tensorflow as tf
    from tensorflow import keras
    import numpy as np
    import matplotlib.pyplot as plt
    return keras, np, plt, tf


@app.cell(hide_code=True)
def _(mo):
    Download_Data = mo.ui.run_button(label="Download Data", kind="info")
    Download_Data
    return (Download_Data,)


@app.cell
def _(Download_Data, keras, mo):
    mo.stop(not Download_Data.value)
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 2. Perform  Eda – check data and labels shape:""")
    return


@app.cell
def _(X_test, X_train):
    # checking images shape
    X_train.shape, X_test.shape
    return


@app.cell
def _(X_train):
    # display single image shape
    X_train[0].shape
    return


@app.cell
def _(y_train):
    # checking labels
    y_train[:5]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 3. Apply Preprocessing: 
        Scaling images(NumPy array) by 255 and One-Hot Encoding labels to represent all categories as 0, except  1 for the actual label in ‘float32.’
        """
    )
    return


@app.cell
def _(X_test, X_train, keras, y_test, y_train):
    # scaling image values between 0-1
    X_train_scaled = X_train/255
    X_test_scaled = X_test/255

    # one hot encoding labels
    # y_train_encoded = keras.utils.to_categorical(y_train, num_classes = 10, dtype = 'float32')
    # y_test_encoded = keras.utils.to_categorical(y_test, num_classes = 10, dtype = 'float32')

    y_train_encoded = keras.utils.to_categorical(y_train, num_classes = 10)
    y_test_encoded = keras.utils.to_categorical(y_test, num_classes = 10)
    return X_test_scaled, X_train_scaled, y_test_encoded, y_train_encoded


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 4. Model Building: 
        A fn to build a neural network with architecture as below with compiling included :
        """
    )
    return


@app.cell
def _(keras, tf):
    def get_model():
        model = keras.Sequential([
            keras.layers.Input(shape=(32,32,3)),
            tf.keras.layers.Flatten(),
            keras.layers.Dense(3000, activation='relu'),
            keras.layers.Dense(1000, activation='relu'),
            keras.layers.Dense(10, activation='sigmoid')    
        ])
        model.compile(optimizer='SGD',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        return model
    return (get_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 5. Training: 
        Train for ten epochs which verbose = 0, meaning no logs.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## CPU
         here -n1 -r1 will ensure the process will run for only one pass, not specifying will perform runs for few no of times and then calculate the average. 

         Also (CPU:0) refers to the first CPU(I have only one).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    Run_CPU_Model_NN = mo.ui.run_button(label="Run CPU Model", kind="danger")
    Run_CPU_Model_NN
    return (Run_CPU_Model_NN,)


@app.cell
def _(
    Run_CPU_Model_NN,
    X_train_scaled,
    get_model,
    mo,
    tf,
    y_train_encoded,
):
    # magic command not supported in marimo; please file an issue to add support
    # %%timeit -n1 -r1
    # CPU
    mo.stop(not Run_CPU_Model_NN.value)
    with tf.device('/CPU:0'):
        model_cpu = get_model()
        model_cpu.fit(X_train_scaled, y_train_encoded, epochs = 5)
    return (model_cpu,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## GPU""")
    return


@app.cell(hide_code=True)
def _(mo):
    Run_GPU_Model_NN = mo.ui.run_button(label="Run GPU Model", kind="danger")
    Run_GPU_Model_NN
    return (Run_GPU_Model_NN,)


@app.cell
def _(
    Run_GPU_Model_NN,
    X_train_scaled,
    get_model,
    mo,
    tf,
    y_train_encoded,
):
    # magic command not supported in marimo; please file an issue to add support
    # %%timeit -n1 -r1
    # GPU
    mo.stop(not Run_GPU_Model_NN.value)
    with tf.device('/GPU:0'):
        model_gpu = get_model()
        model_gpu.fit(X_train_scaled, y_train_encoded, epochs = 10)
    return (model_gpu,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# TEST Two – Training Clothes Classifier""")
    return


@app.cell(hide_code=True)
def _(mo):
    Download_Data_img = mo.ui.run_button(label="Download Data Img", kind="info")
    Download_Data_img
    return (Download_Data_img,)


@app.cell
def _(Download_Data_img, keras, mo):
    mo.stop(not Download_Data_img.value)
    # loading dataset
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # checking shape

    print(train_images.shape)

    print(train_labels[0])
    return (
        fashion_mnist,
        test_images,
        test_labels,
        train_images,
        train_labels,
    )


@app.cell(hide_code=True)
def _(mo, plt, train_images):
    # checking images
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    plt.imshow(train_images[0])
    mo.mpl.interactive(plt.gcf())
    return (class_names,)


@app.cell(hide_code=True)
def _(class_names, train_labels):
    class_names[train_labels[0]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Model Building:""")
    return


@app.cell
def _(keras, test_images, train_images):
    train_images_scaled = train_images / 255.0
    test_images_scaled = test_images / 255.0

    def get_model_1(hidden_layers=1):
        layers = [keras.layers.Flatten(input_shape=(28, 28))]
        for i in range(hidden_layers):
            layers.append(keras.layers.Dense(500, activation='relu'))
        layers.append(keras.layers.Dense(10, activation='sigmoid'))
        model = keras.Sequential(layers)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    return get_model_1, test_images_scaled, train_images_scaled


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If this seems unfamiliar, let me break it down:

        In the above code, we store layers as a list and then append those hidden layers as provided in the hidden_layers. Finally, we compile our model with adam as optimizer and sparce_categorical_crossentropy as loss fn. Metric to monitor is again accuracy.

        Finally, let’s train our model with 5 hidden layers for 5 epochs:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## CPU""")
    return


@app.cell(hide_code=True)
def _(mo):
    Run_CPU_Model = mo.ui.run_button(label="Run CPU Model", kind="danger")
    Run_CPU_Model
    return (Run_CPU_Model,)


@app.cell
def _(
    Run_CPU_Model,
    get_model_1,
    mo,
    tf,
    train_images_scaled,
    train_labels,
):
    mo.stop(not Run_CPU_Model.value)

    # Model Training with CPU
    with tf.device('/CPU:0'):
        cpu_model = get_model_1(hidden_layers=5)
        cpu_model.fit(train_images_scaled, train_labels, epochs=5)

    return (cpu_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# GPU""")
    return


@app.cell(hide_code=True)
def _(mo):
    Run_GPU_Model = mo.ui.run_button(label="Run GPU Model", kind="danger")
    Run_GPU_Model
    return (Run_GPU_Model,)


@app.cell
def _(
    Run_GPU_Model,
    get_model_1,
    mo,
    tf,
    train_images_scaled,
    train_labels,
):
    mo.stop(not Run_GPU_Model.value)

    # Model Training with GPU
    with tf.device('/GPU:0'):
        gpu_model = get_model_1(hidden_layers=5)
        gpu_model.fit(train_images_scaled, train_labels, epochs=5)
    return (gpu_model,)


if __name__ == "__main__":
    app.run()
