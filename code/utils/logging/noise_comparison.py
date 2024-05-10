import os
from pathlib import Path
import numpy as np

def compare_noisy_samples_with_potentially_noisy_samples(noisy_labels, 
        potentially_noisy_samples, logger):

    potentially_noisy_samples_1, potentially_noisy_samples_2 = potentially_noisy_samples
    potentially_noisy_samples_1 = set(potentially_noisy_samples_1.numpy())
    potentially_noisy_samples_2 = set(potentially_noisy_samples_2.numpy())
    noisy_samples = set(noisy_labels[:,0])

    common_potentially_noisy_samples = potentially_noisy_samples_1.intersection(potentially_noisy_samples_2)
    true_guesses_model1 = potentially_noisy_samples_1.intersection(noisy_samples)
    true_guesses_model2 = potentially_noisy_samples_2.intersection(noisy_samples)
    true_guesses_both_models = common_potentially_noisy_samples.intersection(noisy_samples)

    ratio_common_guesses = len(common_potentially_noisy_samples) / len(potentially_noisy_samples_1)

    ratio_true_guesses_to_potentials_model1 = len(true_guesses_model1) / len(potentially_noisy_samples_1)
    ratio_true_guesses_to_potentials_model2 = len(true_guesses_model2) / len(potentially_noisy_samples_2)
    ratio_true_guesses_to_potentials_both_models = len(true_guesses_both_models) / len(common_potentially_noisy_samples)

    ratio_true_guesses_to_noise_model1 = len(true_guesses_model1) / len(noisy_samples)
    ratio_true_guesses_to_noise_model2 = len(true_guesses_model2) / len(noisy_samples)
    ratio_true_guesses_to_noise_both_models = len(true_guesses_both_models) / len(noisy_samples)

    logger.info(f"ratio_true_guesses_to_noise_model1: {ratio_true_guesses_to_noise_model1}")
    logger.info(f"ratio_true_guesses_to_potentials_model1: {ratio_true_guesses_to_potentials_model1}")

    logger.info(f"ratio_true_guesses_to_noise_model2: {ratio_true_guesses_to_noise_model2}")
    logger.info(f"ratio_true_guesses_to_potentials_model2: {ratio_true_guesses_to_potentials_model2}")

    logger.info(f"ratio_true_guesses_to_noise_both_models: {ratio_true_guesses_to_noise_both_models}")
    logger.info(f"ratio_true_guesses_to_potentials_both_models: {ratio_true_guesses_to_potentials_both_models}")

def save_noisy_samples(potentially_noisy_samples, epoch, logname, model):
    np_folder = os.path.join(logname, 'potential_noise')
    Path(np_folder).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(np_folder, f'potentially_noisy_samples_in_epoch_{epoch}_{model}.npy'), 'wb') as f:
        np.save(f, np.array(potentially_noisy_samples))
