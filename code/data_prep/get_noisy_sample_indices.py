import numpy as np
import time

def get_noisy_sample_indices(NUM_TRAINING_SAMPLES, noise_percentage):
    return np.random.choice(range(NUM_TRAINING_SAMPLES), int(NUM_TRAINING_SAMPLES * noise_percentage), replace=False)

def get_noisy_labels_per_noisy_sample(NUM_TRAINING_SAMPLES, NUM_OF_CLASSES, noisy_sample_indices, noise_percentage):
    noisy_labels_per_sample = np.zeros((NUM_TRAINING_SAMPLES, int(NUM_OF_CLASSES*noise_percentage)), dtype=int)

    for i in np.nditer(noisy_sample_indices):
        noisy_labels_per_sample[i] = np.random.choice(range(NUM_OF_CLASSES), size=int(NUM_OF_CLASSES*noise_percentage), replace=False) 

    return noisy_labels_per_sample
