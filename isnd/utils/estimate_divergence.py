import torch
import time
import math
from isnd.model.isnd import IsndMixtureModel
from isnd.model.von_mises.independent_vonMises import IndependentVonMisesMixtureModel
import numpy as np
import gc
from isnd.evaluation.kl_approximation import estimate_kl_samples_density
import os
import json


def select_indices(tensor):
    # List to hold indices of interest
    selected_indices = []
    # Iterate over the tensor in steps of 2
    value_old = 100.0
    n_components_old = 10.0
    for i in range(0, len(tensor)):
        # Compare values in the second column and select index with smaller value
        n_components = tensor[i, 0]
        if n_components == n_components_old:
            if tensor[i, 1] <= value_old:
                index_candidate = i
                value_old = tensor[i, 1]
        else:
            selected_indices.append(index_candidate)
            n_components_old += 10.0
            # print(n_components_old)
            value_old = 100.0
    selected_indices.append(index_candidate)
    return selected_indices


def calculate_kl_estimate(samples_test, model, estimate_kl):
    """Calculate the kl estimates

    Args:
        samples_test: samples from the empirical distribution (test set the model was not trained on)
        model: the model to evaluate

    Returns:
        kl_estimate: the kl_estimate of the model
    """

    # torch.manual_seed(int(time.time()))
    samples_log_density_output = model.log_density(samples_test).detach()
    neg_log_likelihood = -samples_log_density_output.mean()
    print("average log density", samples_log_density_output.mean())
    kl_estimate = None
    if estimate_kl:
        kl_estimate = estimate_kl_samples_density(
            samples_test.to(torch.float32),
            samples_log_density_output.to(torch.float32),
        ).detach()
    return kl_estimate, neg_log_likelihood


def calculate_and_save_estimations(
    test_set,
    save_dir,
    components_list,
    model_paths,
    model_type,
    network_paramterization=True,
):
    """Calculate and store the estimates of kl divergence and sinkhorn distance

    Args:
        test_set: test set the model shall be evaluated on
        save_dir: location to store the estimates
        components_list: number of components for the mixture models
        model_paths: paths to the model
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ###############################################################################################
    # calculate and save kl estimation
    ###############################################################################################
    estimate_kl = True
    kl_estimates = []
    n_dim = test_set.shape[1]
    neg_log_likelihoods = []
    if True:
        with torch.no_grad():
            for n_components, model_path in zip(components_list, model_paths):
                start_time = time.time()
                if model_type == "isnd":
                    device = test_set.device
                    model = IsndMixtureModel(
                        n_dim,
                        n_components,
                        network_paramterization=network_paramterization,
                    )
                    model.to(torch.float64)
                    test_set_adapted = test_set.to(torch.float64)
                    model.to(device)
                elif model_type == "von_mises":
                    device = "cpu"
                    model = IndependentVonMisesMixtureModel(n_dim, n_components)
                    model = model.to(dtype=torch.float32, device=device)
                    test_set_adapted = test_set.to(dtype=torch.float32, device=device)
                model.load_state_dict(torch.load(model_path))
                kl_estimate, neg_log_likelihood = calculate_kl_estimate(
                    test_set_adapted, model, estimate_kl
                )
                kl_estimates.append([n_components, kl_estimate])
                print(f"kl estimate for {model_type}: {kl_estimate}")
                neg_log_likelihoods.append([n_components, neg_log_likelihood])
                end_time = time.time()
                print(
                    f"Time taken for {n_components} components: {(end_time - start_time)/60} minutes"
                )
                del model  # delete the model
                torch.cuda.empty_cache()  # clear unused GPU memory
                gc.collect()
        if estimate_kl:
            kl_estimates_tensor = torch.tensor(kl_estimates)
            kl_file = save_dir + f"{model_type}_ndim_{n_dim}_kl_estimates.pt"
            torch.save(kl_estimates_tensor, kl_file)
        neg_log_likelihood_tensor = torch.tensor(neg_log_likelihoods)
        neg_log_likelihood_file = (
            save_dir + f"{model_type}_ndim_{n_dim}_neg_log_likelihood.pt"
        )
        torch.save(neg_log_likelihood_tensor, neg_log_likelihood_file)


def fetch_test_data(device, data_path, total_set=False):
    """Load data and split it the same way used during training, to use the correct test set.

    Args:
        device: device to load the data on
        data_path: path to the data

    Returns:
        samples_test: test set
    """
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    data = torch.tensor(np.load(data_path)) % (2 * math.pi)
    # shuffle the data:
    ##############################################################
    data = data[torch.randperm(data.shape[0]), :].to(device)
    total_indices = torch.randperm(data.shape[0])
    train_size = int(0.8 * data.shape[0])
    test_indices = total_indices[train_size:]
    samples_test = data[test_indices]
    if total_set:
        return data[total_indices]
    return samples_test


if __name__ == "__main__":
    device = 0
    # load data
    ##############################################################
    components_list = list(range(10, 301, 10))
    model_type = "isnd"
    if model_type == "isnd":
        model_paths = "isnd/config/isnd_checkpoints.json"
    elif model_type == "von_mises":
        model_paths = "isnd/config/von_mises_checkpoints.json"
    with open(model_paths, "r") as file:
        checkpoint_paths = json.load(file)
    ###############################################################################################
    ## calculate estimations for alanine tetrapeptide
    ###############################################################################################
    data_path = "datasets/alaninetetrapeptide/alanine_tetrapeptide.npy"
    samples_test = fetch_test_data(device, data_path)
    save_directory = "isnd/results/alanine_tetrapeptide/"
    base_directory = f"isnd/checkpoints/alanine_tetrapeptide/{model_type}/"
    total_model_dirs_alanine_tetrapeptide = [
        base_directory + model_path
        for model_path in checkpoint_paths["alanine4_15Mar_selected"]
    ]
    calculate_and_save_estimations(
        samples_test,
        save_directory,
        components_list,
        total_model_dirs_alanine_tetrapeptide,
        model_type,
    )
