"""Preprocess the downloaded chignolin data: https://figshare.com/articles/dataset/Chignolin_Simulations/13858898
"""

from multiprocessing import Pool
import mdtraj as md
import os
import fnmatch
import numpy as np


def find_xtc_files(directory):
    xtc_files = []
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, "*.xtc"):
            if not filename.startswith("."):  # Check if the file is not hidden
                xtc_files.append(os.path.join(root, filename))
    return xtc_files


def process_chunk(xtc_paths_chunk, topology_path):
    angles_list = []
    for xtc_path in xtc_paths_chunk:
        trajectory = md.load(xtc_path, top=topology_path)
        non_sodium_selection = trajectory.topology.select("not resname SOD")
        trajectory_without_sodium = trajectory.atom_slice(non_sodium_selection)

        phi = md.compute_phi(trajectory_without_sodium)
        psi = md.compute_psi(trajectory_without_sodium)
        angles = np.hstack((phi[1], psi[1]))

        angles_list.append(angles)

    # Concatenate all angles from the chunk into a single array
    chunk_angles = np.concatenate(angles_list, axis=0)
    return chunk_angles


def chunkify(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def parallel_process(xtc_paths, topology_path, n_processors=4):
    chunks = list(chunkify(xtc_paths, 100))
    with Pool(n_processors) as pool:
        results = pool.starmap(
            process_chunk, [(chunk, topology_path) for chunk in chunks]
        )
    # Concatenate results from all chunks
    all_angles = np.concatenate(results, axis=0)
    return all_angles


if __name__ == "__main__":
    directory = "datasets/chignolin_download"
    xtc_paths = find_xtc_files(directory)
    topology = os.path.join(directory,"filtered","filtered.pdb")
    angles = parallel_process(xtc_paths, topology, n_processors=20)
    save_path =  os.path.join(directory,"chignolin_big.npy")
    np.save(save_path, angles)