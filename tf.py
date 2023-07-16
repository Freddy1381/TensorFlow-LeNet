import tensorflow as tf
print("TensorFlow version: ", tf.__version__)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

INPUT_SHAPE = (28, 28, 1)
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='sigmoid', input_shape=INPUT_SHAPE, padding="same"), 
  tf.keras.layers.AvgPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'), 
  tf.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='sigmoid', padding='valid'), 
  tf.keras.layers.AvgPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'), 
  tf.keras.layers.Flatten(), 
  tf.keras.layers.Dense(120, activation='sigmoid'),
  tf.keras.layers.Dense(84, activation='sigmoid'),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
print("predictions: ", predictions)

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)