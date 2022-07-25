from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf


def load_dataset():
    X, y = make_moons(n_samples=1000, noise=0.25, random_state=100)
    make_plot(X, y, "Classification Dataset Visualization")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def make_plot(x, y, plot_name):
    plt.figure(figsize=(10, 8))
    axes = plt.gca()
    axes.set(xlabel="x", ylabel="y")
    plt.title(plot_name, fontsize=30)
    plt.subplots_adjust(left=0.20)
    plt.subplots_adjust(right=0.80)
    plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), s=40, cmap=plt.cm.Spectral)
    # cmap = plt.cm.Spectral : Points with label 1 are given one color, and points with label 0 are given another color.
    plt.show()
    plt.close()


def network_layers_influence(X_train, y_train):
    print("===============================network_layers_influence==========================")
    for n in range(5):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(8, input_dim=2, activation='relu'))
        for _ in range(n):
            model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=500,verbose=1)
        print("------------------------------------------------------------------------------")
    print("===============================end===================================================")


def dropout_influence(X_train, y_train):
    print("===============================dropout_influence==========================")
    for n in range(5):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(8, input_dim=2, activation='relu'))
        counter = 0
        for _ in range(5):
            model.add(tf.keras.layers.Dense(64, activation='relu'))
        if counter < n:
            counter += 1
            model.add(tf.keras.layers.Dropout(rate=0.5))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=500,verbose=1)
        print("------------------------------------------------------------------------------")
    print("===============================end===================================================")


def build_model_with_regularization(_lambda):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(8, input_dim=2, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(_lambda)))
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(_lambda)))
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(_lambda)))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # 模型装配
    return model


def regularizers_influence(X_train, y_train):
    print("===============================regularizers_influence==========================")
    for _lambda in [1e-5, 1e-3, 1e-1, 0.12, 0.13]:
        model = build_model_with_regularization(_lambda)
        model.fit(X_train, y_train, epochs=500, verbose=1)
        print("------------------------------------------------------------------------------")
    print("===============================end===================================================")


def main():
    X_train, X_test, y_train, y_test = load_dataset()
    network_layers_influence(X_train, y_train)
    dropout_influence(X_train, y_train)
    regularizers_influence(X_train, y_train)

if __name__ == '__main__':
    main()