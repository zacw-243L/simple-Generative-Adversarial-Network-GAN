import tensorflow as tf
import os
import time
import numpy as np
from training_data import *
import seaborn as sb
import matplotlib.pyplot as plt

sb.set()


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


class Generator(tf.keras.Model):
    def __init__(self, hsize=[16, 16], name=None):
        super().__init__(name=name)
        self.dense1 = tf.keras.layers.Dense(hsize[0], activation=tf.nn.leaky_relu)
        self.dense2 = tf.keras.layers.Dense(hsize[1], activation=tf.nn.leaky_relu)
        self.out = tf.keras.layers.Dense(2)

    def call(self, Z):
        h1 = self.dense1(Z)
        h2 = self.dense2(h1)
        out = self.out(h2)
        return out


class Discriminator(tf.keras.Model):
    def __init__(self, hsize=[16, 16], name=None):
        super().__init__(name=name)
        self.dense1 = tf.keras.layers.Dense(hsize[0], activation=tf.nn.leaky_relu)
        self.dense2 = tf.keras.layers.Dense(hsize[1], activation=tf.nn.leaky_relu)
        self.dense3 = tf.keras.layers.Dense(2)
        self.out = tf.keras.layers.Dense(1)

    def call(self, X):
        h1 = self.dense1(X)
        h2 = self.dense2(h1)
        h3 = self.dense3(h2)
        out = self.out(h3)
        return out, h3


generator = Generator()
discriminator = Discriminator()


def compute_loss(X_batch, Z_batch):
    G_sample = generator(Z_batch)
    r_logits, r_rep = discriminator(X_batch)
    f_logits, g_rep = discriminator(G_sample)

    disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(r_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.ones_like(f_logits)))

    return disc_loss, gen_loss, r_rep, g_rep


optimizer = tf.optimizers.RMSprop(learning_rate=0.001)

batch_size = 256
nd_steps = 10
ng_steps = 10

x_plot = sample_data(n=batch_size)

f = open('loss_logs.csv', 'w')
f.write('Iteration,Discriminator Loss,Generator Loss\n')

# Debug print trainable variables
print("Discriminator trainable variables:", discriminator.trainable_variables)
print("Generator trainable variables:", generator.trainable_variables)

generator_optimizer = tf.optimizers.RMSprop(learning_rate=0.001)
discriminator_optimizer = tf.optimizers.RMSprop(learning_rate=0.001)

# Ensure the directories exist
output_dir_iterations = '../GAN/plots/iterations'
output_dir_features = '../GAN/plots/features'
os.makedirs(output_dir_iterations, exist_ok=True)
os.makedirs(output_dir_features, exist_ok=True)

total_start_time = time.time()

# Training loop
for i in range(10001):
    start_time = time.time()  # Start timing
    X_batch = sample_data(n=batch_size)
    Z_batch = sample_Z(batch_size, 2)

    for _ in range(nd_steps):
        with tf.GradientTape() as tape:
            dloss, gloss, r_rep, g_rep = compute_loss(X_batch, Z_batch)
        grads = tape.gradient(dloss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

    for _ in range(ng_steps):
        with tf.GradientTape() as tape:
            dloss, gloss, r_rep, g_rep = compute_loss(X_batch, Z_batch)
        grads = tape.gradient(gloss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

    end_time = time.time()  # End timing
    time_taken = end_time - start_time

    print(f"\rIterations: {i}\t Discriminator loss: {dloss:.4f}\t Generator loss: {gloss:.4f}\t Time taken: {time_taken:.2f} seconds", end="", flush=True)
    if i % 10 == 0:
        f.write("%d,%f,%f\n" % (i, dloss, gloss))

    if i % 1000 == 0:
        plt.figure()
        g_plot = generator(Z_batch)
        xax = plt.scatter(x_plot[:, 0], x_plot[:, 1])
        gax = plt.scatter(g_plot[:, 0], g_plot[:, 1])

        plt.legend((xax, gax), ("Real Data", "Generated Data"))
        plt.title('Samples at Iteration %d' % i)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir_iterations, 'iteration_%d.png' % i))
        plt.close()

        plt.figure()
        rrd = plt.scatter(r_rep[:, 0], r_rep[:, 1], alpha=0.5)
        rrg = plt.scatter(g_rep[:, 0], g_rep[:, 1], alpha=0.5)

        plt.legend((rrd, rrg), ("Real Data Representation", "Generated Data Representation"))
        plt.title('Transformed Features at Iteration %d' % i)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir_features, 'feature_transform_%d.png' % i))
        plt.close()

        plt.figure()

        rrdc = plt.scatter(np.mean(r_rep[:, 0]), np.mean(r_rep[:, 1]), s=100, alpha=0.5)
        rrgc = plt.scatter(np.mean(g_rep[:, 0]), np.mean(g_rep[:, 1]), s=100, alpha=0.5)

        plt.legend((rrdc, rrgc), ("Real Data Representation Centroid", "Generated Data Representation Centroid"))

        plt.title('Centroid of Transformed Features at Iteration %d' % i)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir_features, 'feature_transform_centroid_%d.png' % i))
        plt.close()

total_end_time = time.time()
total_time_taken = total_end_time - total_start_time
print(f"\nTotal time taken for all iterations: {total_time_taken:.2f} seconds")

f.close()
