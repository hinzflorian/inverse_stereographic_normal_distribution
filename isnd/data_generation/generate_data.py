"""
Generate torsion data for alanine tetrapeptide and chignolin.
"""
# env: isnd_code3
import os
import bgmol
import mdtraj as md
import numpy as np
import openmm
import torch
from bgflow.utils.types import assert_numpy
from openmm import unit
###################################################################################################
# define global variables
###################################################################################################
kB = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)
###################################################################################################
# define function to get bgmol model
###################################################################################################
def get_model(system_name, temperature=None):
    """Get the bgmol model for the given system name.

    Args:
        system_name: the system to run simulations for. E.g. AlanineTetrapeptideVacuum
        temperature: the temperature at which simulations were run

    Returns:
        model: corresponding bgmol model for system
    """
    model = bgmol.system_by_name(system_name.replace("ModifiedPSI", ""))
    if "ModifiedPSI" in system_name:
        extraBias_str = "100*sin(0.5*theta)^2"
        extraBias = openmm.CustomTorsionForce(extraBias_str)
        psi_angles = md.compute_psi(
            md.Trajectory(model.positions, model.mdtraj_topology)
        )[0]
        for i in range(len(psi_angles)):
            extraBias.addTorsion(*psi_angles[i])
            print(f"{system_name}, adding bias on psi{psi_angles[i]}: {extraBias_str}")
        model.system.addForce(extraBias)
    if temperature is not None:
        model.reinitialize_energy_model(temperature=temperature)
    return model
###################################################################################################
# Function for running MD
###################################################################################################
def run_traj(input_temp, system_config_name, output_dir, n_iter):
    """Runs md simulations, saves and returns the trajectory data.

    Args:
        input_temp: temperature at which to run the simulations
        system_config_name: the system to run simulations for. E.g. AlanineTetrapeptideVacuum
        output_dir: directory to save md trajectories
        n_iter: number of iterations for md simulations

    Returns:
        md_trajectory: md trajectory
        energy_values: corresponding energy values
    """
    model = get_model(system_config_name)
    prior_filepath = os.path.join(output_dir,f"MD-{system_config_name}-T{input_temp}.npz")
    try:
        # load MD data if available
        MDdata = np.load(prior_filepath)["data"]
        MDener = np.load(prior_filepath)["ener"]
    except:
        # otherwise run the simulations
        temp = input_temp
        pace = 500
        n_equil = 100

        integrator = openmm.LangevinMiddleIntegrator(
            temp * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtosecond
        )
        simulation = openmm.app.Simulation(model.topology, model.system, integrator)
        simulation.context.setPositions(model.positions)
        MDdata = np.full((n_iter, *model.positions.shape), np.nan)
        MDener = np.full(n_iter, np.nan)
        # equilibrate
        print("equilibrating...")
        simulation.step(pace * n_equil)
        # run MD
        for n in range(n_iter):
            simulation.step(pace)
            MDdata[n] = (
                simulation.context.getState(getPositions=True)
                .getPositions()
                .value_in_unit(unit.nanometer)
            )
            MDener[n] = (
                simulation.context.getState(getEnergy=True)
                .getPotentialEnergy()
                .value_in_unit(unit.kilojoule_per_mole)
            )
            print(f"sampling...  {(n+1)/n_iter:.1%}", end="\r")
        np.savez(prior_filepath, data=MDdata, ener=MDener)
    md_trajectory = (
        torch.tensor(MDdata).view(-1, 3 * len(model.positions)).to(dtype=torch.float32)
    )
    energy_values = (
        torch.tensor(MDener / (kB * input_temp)).view(-1, 1).to(dtype=torch.float32)
    )
    del MDdata, MDener
    return md_trajectory, energy_values


def convert_to_torsion_data(system_config_name, trajectory_data, save_path):
    """Convert the trajectory data to torsion data and save it to a file.

    Args:
        system_config_name: name and condition of the system
        trajectory_data: the md trajectory data
        save_path: path to file for saving the torsion data
    """
    model = get_model(system_config_name)
    trajectory = assert_numpy(
        trajectory_data.view(len(trajectory_data), *model.positions.shape)
    )
    trajectory = md.Trajectory(trajectory, model.mdtraj_topology)
    phi = md.compute_phi(trajectory)
    psi = md.compute_psi(trajectory)
    # Stack phi and psi arrays together
    angles = np.hstack((phi[1], psi[1]))
    np.save(save_path, angles)


def generate_torsion_data(
    temperature, system_config_name, md_data_dir, n_iter, torsion_data_save_path
):
    """Run md simulations and convert the trajectory data to torsion data.

    Args:
        temperature: the temperature at which simulations were run
        system_config_name: the system to run simulations for. E.g. AlanineTetrapeptideVacuum
        md_data_dir: directory to save md trajectories
        n_iter: number of iterations to run MD simulations
        torsion_data_save_path: path to file for saving the torsion data
    """
    if not os.path.exists(md_data_dir):
        os.makedirs(md_data_dir)

    trajectory_data, _ = run_traj(
        temperature, system_config_name, md_data_dir, n_iter
    )  # energies are reduced units
    convert_to_torsion_data(system_config_name, trajectory_data, torsion_data_save_path)


###################################################################################################
if __name__ == "__main__":
    T_low = 300
    ###############################################################################################
    # for alanine tetrapeptide
    ###############################################################################################
    #n_iter = 2000000
    n_iter = 200
    system_name = "AlanineTetrapeptide"
    system_config_name = system_name + "Vacuum"
    output_directory = "./datasets/alaninetetrapeptide/"
    torsion_data_save_path = os.path.join(output_directory,"alanine_tetra_peptide_torsions_long.npy")
    generate_torsion_data(
        T_low, system_config_name, output_directory, n_iter, torsion_data_save_path
    )
    ###############################################################################################
    # for chignolin
    ###############################################################################################
    #n_iter = 2000000
    system_config_name = "ChignolinC22Implicit"
    output_directory = "./datasets/chignolin"
    torsion_data_save_path =os.path.join(output_directory,"chignolin.npy")
    generate_torsion_data(
        T_low, system_config_name, output_directory, n_iter, torsion_data_save_path
    )