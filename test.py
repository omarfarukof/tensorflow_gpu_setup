import tensorflow as tf
import time

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# # Define the model
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])

# # Define the loss function, optimizer, and metrics
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# optimizer = tf.keras.optimizers.Adam()
# metrics = ['accuracy']

# # Compile the model
# model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

# Run on CPU
with tf.device('/CPU:0'):

    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Define the loss function, optimizer, and metrics
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
    metrics = ['accuracy']

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    print("Running on CPU:")
    start_time = time.time()
    model.fit(x_train, y_train, epochs=5)
    end_time = time.time()
    cpu_train_time = end_time - start_time
    print(f"CPU training time: {cpu_train_time:.2f} seconds")
    start_time = time.time()
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    end_time = time.time()
    cpu_test_time = end_time - start_time
    print(f"CPU testing time: {cpu_test_time:.2f} seconds")
    print(f"CPU test accuracy: {test_acc:.2f}")

# Run on GPU
with tf.device('/GPU:0'):

    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Define the loss function, optimizer, and metrics
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
    metrics = ['accuracy']

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    print("\nRunning on GPU:")
    start_time = time.time()
    model.fit(x_train, y_train, epochs=5)
    end_time = time.time()
    gpu_train_time = end_time - start_time
    print(f"GPU training time: {gpu_train_time:.2f} seconds")
    start_time = time.time()
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    end_time = time.time()
    gpu_test_time = end_time - start_time
    print(f"GPU testing time: {gpu_test_time:.2f} seconds")
    print(f"GPU test accuracy: {test_acc:.2f}")

# Print the timing comparison
print(f"\nTiming comparison:")
print(f"CPU training time: {cpu_train_time:.2f} seconds")
print(f"GPU training time: {gpu_train_time:.2f} seconds")
print(f"Speedup: {cpu_train_time / gpu_train_time:.2f}x")
print(f"CPU testing time: {cpu_test_time:.2f} seconds")
print(f"GPU testing time: {gpu_test_time:.2f} seconds")
print(f"Speedup: {cpu_test_time / gpu_test_time:.2f}x")