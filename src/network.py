import tensorflow as tf

def network(problem, m, N_points):
        branch = tf.keras.Sequential(
            [
                
                tf.keras.layers.InputLayer(input_shape=(173*47*200)),
                tf.keras.layers.Reshape((173,47,200)),
                # tf.keras.layers.transpose,
                tf.keras.layers.Conv2D(200, (3, 3), strides=1, activation="ReLU"),
                tf.keras.layers.Conv2D(200, (3, 3), strides=1, activation="ReLU"),
                tf.keras.layers.Conv2D(500, (3, 3), strides=1, activation="ReLU"),
                tf.keras.layers.Conv2D(1000, (3, 3), strides=1, activation="ReLU"),
                tf.keras.layers.Dropout(0.5),
                # tf.keras.layers.Conv2D(1000, (1, 1), strides=1, activation="ReLU"),
                # tf.keras.layers.Conv2D(500, (1, 1), strides=1, activation="ReLU"),
                tf.keras.layers.Conv2D(100, (1, 1), strides=1, activation="ReLU"),
                tf.keras.layers.Conv2D(10, (1, 1), strides=1, activation="ReLU"),
                tf.keras.layers.Conv2D(1, (1, 1), strides=1, activation="ReLU"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1000, activation="ReLU"),
                tf.keras.layers.Dense(1000, activation="ReLU"),
                tf.keras.layers.Dropout(0.5),
                # tf.keras.layers.Dense(1000, activation="ReLU"),
                # tf.keras.layers.Dense(1000, activation="ReLU"),
                tf.keras.layers.Dense(500, activation="ReLU"),
                tf.keras.layers.Dense(200, activation="ReLU"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(47*50),
              
                tf.keras.layers.Reshape((47,50)),
                # tf.keras.layers.Lambda(lambda x: transpose_last_two_dims(x))

            ]
        )
        branch.summary()
        branch = [m, branch]
        trunk=[1,128,256,512,512,1024,2048,4096,2048,50*47]
        dot = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(87 * 24, 2)),
                tf.keras.layers.Flatten(),
            ]
        )
        dot.summary()
        dot = [0, dot]
        return branch, trunk, dot
