import numpy as np
import matplotlib
# Use non-interactive backend for saving figures/animations
# matplotlib.use('Agg') # Commented out for potential interactive use in some environments
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import os
import json
import scipy.sparse
import scipy.sparse.csgraph
import time # Import time for performance measurement
import collections # Import collections for defaultdict
import random # Import random for brain simulation
import math # Import math for brain simulation
import hashlib # Import hashlib for brain simulation
from collections import defaultdict, deque # Import deque for brain simulation
from dataclasses import dataclass, field # Import dataclass and field for brain simulation
from typing import List, Dict, Tuple, Any, Optional # Import typing hints
from scipy.ndimage import gaussian_filter1d # For smoothing activity in brain simulation

# For MAUS integration:
import scipy.linalg as sla # For dense linear algebra (e.g., np.linalg.solve)
import scipy.sparse as sp # For sparse matrix objects
import scipy.sparse.linalg as spla # For sparse linear algebra solvers (e.g., GMRES, sparse SVD)
import cmath
from enum import Enum


# ##########################################################################
# Coupled Particle and Brain Simulation
# ##########################################################################
# This script couples:
# 1. A multi-patch particle simulation with various potentials and dynamics.
# 2. A Unified Brain Simulation (Eigenspace Matrix Mind + Neural Substrate).
#
# Core Idea:
# - The simulation space is divided into conceptual regions (Coupled Zones).
# - Particle density and dynamics in a region influence its Computational Consciousness Zone (CCZ).
# - Each CCZ is controlled by a higher-level adaptive system (MAUS-inspired "Master Mind")
#   that adjusts local parameters (e.g., friction, neurogenesis rates) based on local "stress".
# - The Matrix Mind generates thought sequences targeting brain regions.
# - Activation of a brain region from a thought sequence applies a force to particles
#   in the corresponding spatial region.
# ##########################################################################


# --- Constants & Default Simulation Parameters ---
DEFAULT_EPSILON = 1e-9
MAX_RECENT_SPIKES_PER_NEURON = 50 # Max spikes to store per neuron for STDP lookup

# --- Default Parameters for Particle Simulation ---
default_particle_parameters = {
    "simulation": {
        "num_steps": 100, # Reduced for faster execution/demo
        "dt": 0.1,
        "damping_factor": 0.05,
        "max_velocity": 500.0,
        "animation_fps": 10 # Reduced for faster GIF generation
    },
    "forces": {
        "k_matrix": [[1.0, -0.5], [-0.5, 0.8]], # For center-center pairwise (if used)
        "G_matrix": [[0.1, 0.2], [0.2, 0.05]], # For center-center pairwise (if used)
        "C": 0.01, # Central force constant
        "cutoff_distance": 5.0, # Cutoff for center-center pairwise (if used)
        "short_range_repulsion_strength": 50.0, # For center-center SR (if used)
        "moment_of_inertia_mapping": {"0": 1.0, "1": 2.0, "2": 1.5},
        "patch_params": {
             "enabled": True,
             "patch_definitions": {
                  "0": [
                       {"distance": 1.0, "angle_relative_to_particle": 0.0, "patch_type": 0},
                       {"distance": 1.0, "angle_relative_to_particle": np.pi, "patch_type": 0}
                  ],
                  "1": [
                       {"distance": 1.2, "angle_relative_to_particle": 0.0, "patch_type": 1}
                  ],
                  "2": [
                       {"distance": 1.0, "angle_relative_to_particle": 0, "patch_type": 2},
                       {"distance": 1.0, "angle_relative_to_particle": 2*np.pi/3, "patch_type": 2},
                       {"distance": 1.0, "angle_relative_to_particle": 4*np.pi/3, "patch_type": 2}
                  ]
             },
             # --- Potential Type and Parameters for Patch-Patch Interactions ---
             "patch_pairwise_potential": {
                  "type": "inverse_square_plus_sr", # Options: "inverse_square_plus_sr", "lennard_jones", "square_well"
                  "inverse_square_strength_matrix": [ # Used if type is "inverse_square_plus_sr"
                       [ 10.0,          -5.0,          2.0],
                       [-5.0,           8.0,         -1.0],
                       [ 2.0,          -1.0,          5.0]
                  ],
                  "sr_strength": 100.0, # Used if type is "inverse_square_plus_sr"
                  "lennard_jones": { # Used if type is "lennard_jones"
                       "epsilon_matrix": [
                            [ 1.0,          0.5,          0.2],
                            [ 0.5,          0.8,          0.1],
                            [ 0.2,          0.1,          0.5]
                       ],
                       "sigma_matrix": [
                            [ 1.0,          1.1,          1.2],
                            [ 1.1,          1.0,          1.1],
                            [ 1.2,          1.1,          1.0]
                       ],
                       "cutoff_factor": 2.5
                  },
                  # --- Square Well parameters for pairwise (with transition_width) ---
                  "square_well": { # Used if type is "square_well"
                       "epsilon_matrix": [
                            [ 1.0,          0.5,          0.2],
                            [ 0.5,          0.8,          0.1],
                            [ 0.2,          0.1,          0.5]
                       ],
                       "sigma_matrix": [
                            [ 1.0,          1.1,          1.2],
                            [ 1.1,          1.0,          1.1],
                            [ 1.2,          1.1,          1.0]
                       ],
                       "lambda_matrix": [
                            [ 1.5,          1.6,          1.7],
                            [ 1.6,          1.5,          1.6],
                            [ 1.7,          1.6,          1.5]
                       ],
                       "transition_width": 0.1
                  }
             },
             "patch_cutoff_distance": 3.0 # Still needed as a general cutoff for finding neighboring patches
        },
        "orientation_potential": {
             "bond_angle_potential": {
                  "enabled": False,
                  "strength": 10.0,
                  "ideal_angle_mapping": {"0": 0.0, "1": np.pi}
             }
        }
    },
    "bonding": {
        "enabled": True,
        # --- Potential Type and Parameters for Bonded Patch Interactions ---
        "patch_bond_potential": {
             "type": "harmonic", # Options: "harmonic", "lennard_jones", "square_well"
             "harmonic": { # Used if type is "harmonic"
                  "bond_distance": 1.5,
                  "bond_strength": 200.0
             },
             "lennard_jones": { # Used if type is "lennard_jones" for bonded patches
                  "epsilon": 5.0,
                  "sigma": 1.5
             },
             # --- Square Well parameters for bonded (with transition_width) ---
             "square_well": { # Used if type is "square_well" for bonded patches
                  "epsilon": 5.0,
                  "sigma": 1.5,
                  "lambda": 1.5,
                  "transition_width": 0.1
             }
        },
        "bond_break_distance": 2.0,
        "bond_types": [0, 1],
        "formation_criteria": {
             "distance_tolerance": 0.2,
             "patch_type_compatibility_matrix": [
                  [ True,          False,        False],
                  [ False,         True,         False],
                  [ False,         False,        True]
             ],
             "orientation_alignment_tolerance": np.pi / 4
        }
    },
    "density_repulsion": {
        "density_radius": 10.0,
        "density_repulsion_strength": 5.0
    },
    "boundaries": {
        "x_min": 0,
        "x_max": 100,
        "y_min": 0,
        "y_max": 100
    },
    "saving": {
        "load_simulation_state": False,
        "load_directory": "simulation_state",
        "save_directory": "simulation_state",
        "periodic_save_interval": 100,
        "animation_filename": "particle_simulation_periodic.gif"
    },
     "initial_conditions": {
        "type": "grid_swirl",
        "num_particles_request": 20, # Reduced for faster execution/demo
        "swirl_strength": 0.5,
        "mass_mapping": {"0": 1.0, "1": 2.0, "2": 1.5},
        "initial_orientation_type": "random",
        "initial_orientation_angle": 0.0,
        "new_particle_angular_initialization": {
             "orientation_type": "copy_parent",
             "orientation_angle": 0.0,
             "angular_velocity_type": "copy_scaled_parent",
             "angular_velocity_scale": 1.0
         }
     },
     "analysis": {
         "rdf_dr": 0.5,
         "rdf_rmax": 30.0,
         "rdf_start_frame": 0
     },
    "external_force": {
        "enabled": False, # This will be controlled by the brain now
        "force_vector": [0.0, 0.0], # Initial force vector
        "friction_enabled": True, # Always true when controlled by adaptive factors
        "friction_coefficient": 0.1, # Base friction coefficient
        "external_torque_enabled": False,
        "torque_value": 0.0,
        "angular_friction_enabled": True, # Always true when controlled by adaptive factors
        "angular_friction_coefficient": 0.05 # Base angular friction coefficient
    },
    "state_change": {
        "enabled": False,
        "on_bond_form": {
             "from_type": 0,
             "to_type": 2
        }
    },
    "particle_creation": {
        "enabled": False,
        "creation_rate": 0.01,
        "trigger": {
             "type": 0,
             "min_neighbors": 3
        },
        "new_particle": {
             "type": 2,
             "initial_velocity_scale": 0.1,
             "angular_initialization": {
                 "orientation_type": "copy_parent",
                 "orientation_angle": 0.0,
                 "angular_velocity_type": "copy_scaled_parent",
                 "angular_velocity_scale": 1.0
             }
        }
    },
    "particle_deletion": {
        "enabled": False,
        "deletion_rate": 0.01,
        "trigger": {
             "type": None,
             "condition": "out_of_bounds",
             "buffer_distance": 5.0
        }
    },
    "adaptive_interactions": {
        "enabled": False,
        "bond_strength_adaptation": {
             "enabled": False,
             "adaptation_rate": 0.01,
             "target_strength": 200.0,
             "trigger": "bond_age"
        },
        "pairwise_adaptation": {
             "enabled": False,
             "k_adaptation_rate": 0.001,
             "k_target_matrix": [[2.0, 0.0], [0.0, 1.5]],
             "G_adaptation_rate": 0.0005,
             "G_target_matrix": [[0.05, 0.1], [0.1, 0.01]],
             "trigger": "time"
        }
    },
    "visualization": {
         "orientation_line": {
              "length": 2.0,
              "color": 'black',
              "linewidth": 1.0
         },
         "patches": {
              "enabled": True,
              "size": 0.5,
              "color_mapping": {
                   "0": 'red',
                   "1": 'green',
                   "2": 'purple'
              },
              "edgecolor": 'black'
         },
         "bonds": {
              "enabled": True,
              "color": 'gray',
              "linewidth": 2.0,
              "linestyle": '-'
         },
         "clusters": {
              "enabled": True,
              "color_by": "label",
              "colormap": "viridis"
         }
    }
}

# --- Default Parameters for Unified Brain Simulation ---
@dataclass
class MatrixMindHyperParameters:
    LATENT_DIM: int = 8 # Dimensionality D of latent space & M_context.
                        # This will also be the number of "conceptual regions" in the 1D brain.
    CONTEXT_MATRIX_ETA: float = 0.1 # Update rate for M_context from co-activity (0 < eta <= 1)
    TEMPERATURE: float = 0.9      # Base sampling temperature for softmax in token selection
    NOVELTY_BIAS_STRENGTH: float = 0.25 # How much to favor novel eigenvectors
    EIGENVECTOR_COHERENCE_STRENGTH: float = 0.15 # How much to favor eigenvectors aligned with previous direction
    INTENTION_STRENGTH: float = 0.4    # Influence of external intention vector on eigenvector selection
    MIN_EIGENVALUE_THRESHOLD: float = 1e-5 # Ignore eigenvectors with very small eigenvalues
    NUM_EIGENVECTORS_TO_CONSIDER: int = 3 # Consider top N eigenvectors for token scoring
    N_STEPS_PER_THOUGHT_SEQUENCE: int = 10 # Number of steps in a "thought sequence"
    RANDOM_JUMP_PROBABILITY: float = 0.02 # Chance to randomly perturb M_context

default_matrix_mind_params = MatrixMindHyperParameters()

# Grid Parameters for Brain Substrate (aligned with particle simulation regions)
# L_SUBSTRATE will be the number of regions
# NX_SUBSTRATE will be the number of regions
# DX_SUBSTRATE will be 1.0
# X_GRID_SUBSTRATE will be np.arange(num_conceptual_regions)

# Time Parameters for Brain Simulation
brain_time_params = {
    'T_FINAL_SIMULATION': 20.0, # Total simulation time (will match particle sim time)
    'DT_OUTER_LOOP': default_particle_parameters["simulation"]["dt"], # Brain updates at particle sim dt
    'DT_INNER_LOOP_NEURAL': 0.01, # Time step for neural dynamics ( finer scale)
    'PROGRESS_PRINT_INTERVAL': 1 # Print progress every N outer steps
}

# Substrate PDE Parameters (simplified for 1D regions)
default_params_substrate_pde = {
    'D_E': 0.01, 'k_E': 0.1, 'rho_E': 0.05, 'Q_E_baseline': 0.0, # Neurotrophic E
    'D_S': 0.001, 'k_S': 0.05, 'S_0': 1.0, 'gamma_S': 0.01, 'Q_S_baseline': 0.0, # Structure S
    'D_M': 0.02, 'k_M': 0.2, 'rho_M': 0.1, 'Q_M_baseline': 0.0  # Modulator M
}

# Neuron Parameters (Izhikevich Model)
default_params_neuron_model = {
    'v_peak': 30.0,
    'a0': 0.02, 'b0': 0.2, 'c0': -65.0, 'd0': 8.0,
    # These functions will now take region index as input to get field values
    'f_a': lambda E, S: default_params_neuron_model['a0'],
    'f_b': lambda E, S: default_params_neuron_model['b0'],
    'f_c': lambda E, S: default_params_neuron_model['c0'] - 2 * E + 5 * max(0, S - 0.8),
    'f_d': lambda E, S: default_params_neuron_model['d0'] + 1 * E - 2 * max(0, S - 0.8),
    'eta_M_modulatory_current': 5.0 # Effect of modulator M on neuron input current
}

# Neurogenesis Parameters
default_params_neurogenesis = {
    'alpha_proliferation_rate': 0.05, # Scaler for proliferation
    'N_max_total_neurons': 50, # Reduced for faster execution/demo
    'initial_N_neurons': 10 # Reduced for faster execution/demo
}

# Synaptogenesis Parameters
default_params_synaptogenesis = {
    'prob_formation_scale': 0.03,
    # Max connection distance will be in terms of region index difference
    'max_connection_region_dist': 2, # Max region difference for connection
    'S_threshold_for_connection': 0.5,
    'E_factor_on_connection_prob': 0.4,
    'initial_weight_mean': 0.5,
    'initial_weight_std': 0.1,
    'tau_synaptic_variable_decay': 5.0 * brain_time_params['DT_INNER_LOOP_NEURAL'] # Decay time for s_j in I = sum w_ij * s_j
}

# STDP Plasticity Parameters
default_params_stdp = {
    'max_dt_time_window': 20.0 * brain_time_params['DT_INNER_LOOP_NEURAL'],
    'tau_plus_potentiation': 5.0 * brain_time_params['DT_INNER_LOOP_NEURAL'],
    'tau_minus_depression': 5.5 * brain_time_params['DT_INNER_LOOP_NEURAL'], # Slightly different for asymmetry
    'A_plus0_base_potentiation': 0.008,
    'A_minus0_base_depression': 0.009,
    # These functions will take region index as input to get field values
    'f_Aplus_substrate_modulation': lambda E, M, S: default_params_stdp['A_plus0_base_potentiation'] * max(0.1, E) * (1 + 0.3*M) * max(0.1, S),
    'f_Aminus_substrate_modulation': lambda E, M, S: default_params_stdp['A_minus0_base_depression'] * max(0.1, E) * (1 + 0.3*M) * max(0.1, S),
    'w_max_synaptic_weight': 8.0,
    'w_min_synaptic_weight': DEFAULT_EPSILON # Effectively prunes if it goes to zero
}

# Activity Smoothing Parameters for field A
default_params_activity_field = {
    'kernel_sigma_spatial_smoothing_regions': 1.5, # Width of Gaussian smoothing for activity across regions
    'tau_A_temporal_decay': brain_time_params['DT_OUTER_LOOP'] * 1.5 # Time constant for activity field A decay
}

# Unified System Parameters (Coupling Parameters)
default_params_unified = {
    'num_conceptual_regions': default_matrix_mind_params.LATENT_DIM, # K regions, must match LATENT_DIM
    'matrix_mind_update_interval': 5, # Update MatrixMind every N outer steps
    'structural_influence_eta': 0.05, # Learning rate for structural influence on M_context
    'thought_application_duration': 3, # Apply stimulation from thought for N outer steps
    'stimulation_strength_from_thought': 5.0, # External current applied to targeted regions in brain
    'particle_density_to_brain_activity_scale': 0.1, # How much particle density influences brain activity field
    'brain_stimulation_to_particle_force_scale': 0.5 # How much brain stimulation influences particle force
}


# --- Function to Load Simulation Parameters from File ---
def load_parameters(filepath="parameters.json"):
    """
    Loads simulation parameters from a JSON file, merging with defaults.
    Handles parameters for both particle and brain simulations.
    MODIFIED: Always returns default parameters, ignoring file system.
    """
    # Start with a copy of default parameters for both systems
    parameters = {
        "particle": default_particle_parameters.copy(),
        "brain": {
            "matrix_mind": default_matrix_mind_params.__dict__.copy(),
            "time": brain_time_params.copy(),
            "substrate_pde": default_params_substrate_pde.copy(),
            "neuron_model": default_params_neuron_model.copy(),
            "neurogenesis": default_params_neurogenesis.copy(),
            "synaptogenesis": default_params_synaptogenesis.copy(),
            "stdp": default_params_stdp.copy(),
            "activity_field": default_params_activity_field.copy(),
            "unified": default_params_unified.copy()
        }
    }

    # --- Post-loading adjustments and type conversions for Particle Params ---
    # These adjustments are necessary even with default parameters, as some defaults
    # (e.g., matrices, mappings) need type conversion (e.g., list to np.ndarray, string keys to int).
    # Since we are returning defaults, we explicitly do this on the defaults to ensure consistency.
    particle_params = parameters["particle"]
    if 'initial_conditions' in particle_params and 'mass_mapping' in particle_params['initial_conditions']:
         particle_params['initial_conditions']['mass_mapping'] = {int(k): v for k, v in particle_params['initial_conditions']['mass_mapping'].items()}
    if 'bonding' in particle_params and 'bond_types' in particle_params['bonding']:
         particle_params['bonding']['bond_types'] = tuple(particle_params['bonding']['bond_types'])
    if 'forces' in particle_params and 'moment_of_inertia_mapping' in particle_params['forces']:
         particle_params['forces']['moment_of_inertia_mapping'] = {int(k): v for k, v in particle_params['forces']['moment_of_inertia_mapping'].items()}
    if 'forces' in particle_params and 'patch_params' in particle_params['forces']:
         if 'patch_definitions' in particle_params['forces']['patch_params']:
              particle_params['forces']['patch_params']['patch_definitions'] = {int(k): v for k, v in particle_params['forces']['patch_params']['patch_definitions'].items()}
    if 'forces' in particle_params and 'orientation_potential' in particle_params['forces']:
         if 'bond_angle_potential' in particle_params['forces']['orientation_potential'] and 'ideal_angle_mapping' in particle_params['forces']['orientation_potential']['bond_angle_potential']['ideal_angle_mapping']:
              particle_params['forces']['orientation_potential']['bond_angle_potential']['ideal_angle_mapping'] = {int(k): v for k, v in particle_params['forces']['orientation_potential']['bond_angle_potential']['ideal_angle_mapping'].items()} # Convert keys to int

    if 'bonding' in particle_params and 'formation_criteria' in particle_params['bonding'] and 'patch_type_compatibility_matrix' in particle_params['bonding']['formation_criteria']:
         # Check if it's already a numpy array from default, otherwise convert
         if not isinstance(particle_params['bonding']['formation_criteria']['patch_type_compatibility_matrix'], np.ndarray):
            particle_params['bonding']['formation_criteria']['patch_type_compatibility_matrix'] = np.array(particle_params['bonding']['formation_criteria']['patch_type_compatibility_matrix'], dtype=bool)

    if 'visualization' in particle_params and 'patches' in particle_params['visualization'] and 'color_mapping' in particle_params['visualization']['patches']:
         particle_params['visualization']['patches']['color_mapping'] = {int(k): v for k, v in particle_params['visualization']['patches']['color_mapping'].items()}

    # Convert matrix parameters within potential definitions to numpy arrays
    if 'forces' in particle_params and 'patch_params' in particle_params['forces'] and 'patch_pairwise_potential' in particle_params['forces']['patch_params']:
         pairwise_potential_params = particle_params['forces']['patch_params']['patch_pairwise_potential']
         if 'inverse_square_strength_matrix' in pairwise_potential_params and not isinstance(pairwise_potential_params['inverse_square_strength_matrix'], np.ndarray):
              pairwise_potential_params['inverse_square_strength_matrix'] = np.array(pairwise_potential_params['inverse_square_strength_matrix'])
         if 'lennard_jones' in pairwise_potential_params:
              lj_params = pairwise_potential_params['lennard_jones']
              if 'epsilon_matrix' in lj_params and not isinstance(lj_params['epsilon_matrix'], np.ndarray):
                   lj_params['epsilon_matrix'] = np.array(lj_params['epsilon_matrix'])
              if 'sigma_matrix' in lj_params and not isinstance(lj_params['sigma_matrix'], np.ndarray):
                   lj_params['sigma_matrix'] = np.array(lj_params['sigma_matrix'])
         if 'square_well' in pairwise_potential_params:
              sw_params = pairwise_potential_params['square_well']
              if 'epsilon_matrix' in sw_params and not isinstance(sw_params['epsilon_matrix'], np.ndarray):
                   sw_params['epsilon_matrix'] = np.array(sw_params['epsilon_matrix'])
              if 'sigma_matrix' in sw_params and not isinstance(sw_params['sigma_matrix'], np.ndarray):
                   sw_params['sigma_matrix'] = np.array(sw_params['sigma_matrix'])
              if 'lambda_matrix' in sw_params and not isinstance(sw_params['lambda_matrix'], np.ndarray):
                   sw_params['lambda_matrix'] = np.array(sw_params['lambda_matrix'])

    # --- Post-loading adjustments and type conversions for Brain Params ---
    brain_params = parameters["brain"]
    # Ensure LATENT_DIM and num_conceptual_regions match
    brain_params["unified"]["num_conceptual_regions"] = brain_params["matrix_mind"]["LATENT_DIM"]
    # Update brain time parameters based on particle simulation dt
    brain_params["time"]["DT_OUTER_LOOP"] = particle_params["simulation"]["dt"]
    brain_params["time"]["T_FINAL_SIMULATION"] = particle_params["simulation"]["num_steps"] * particle_params["simulation"]["dt"]

    # Ensure neuron model functions are callable
    # These lambda definitions rely on `default_params_neuron_model` being available directly,
    # which it is, as `parameters` is initialized from defaults.
    brain_params["neuron_model"]['f_a'] = lambda E, S: brain_params["neuron_model"]['a0']
    brain_params["neuron_model"]['f_b'] = lambda E, S: brain_params["neuron_model"]['b0']
    brain_params["neuron_model"]['f_c'] = lambda E, S: brain_params["neuron_model"]['c0'] - 2 * E + 5 * max(0, S - 0.8)
    brain_params["neuron_model"]['f_d'] = lambda E, S: default_params_neuron_model['d0'] + 1 * E - 2 * max(0, S - 0.8)
    brain_params["neuron_model"]['eta_M_modulatory_current'] = 5.0 # Ensure this specific value is retained and not re-evaluated as part of the lambda.


    # Ensure STDP modulation functions are callable
    # Similarly for STDP lambdas
    brain_params["stdp"]['f_Aplus_substrate_modulation'] = lambda E, M, S: brain_params["stdp"]['A_plus0_base_potentiation'] * max(0.1, E) * (1 + 0.3*M) * max(0.1, S)
    brain_params["stdp"]['f_Aminus_substrate_modulation'] = lambda E, M, S: brain_params["stdp"]['A_minus0_base_depression'] * max(0.1, E) * (1 + 0.3*M) * max(0.1, S)


    print(f"Using default parameters, ignoring '{filepath}'.")
    return parameters


# --- Utility Functions (from both scripts, adapted) ---
def stable_hash(text: str) -> float:
    h = hashlib.md5(text.encode()).hexdigest() # md5 is faster for this purpose
    return int(h, 16) / 2**128

def cosine_similarity_np(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    if not isinstance(vec_a, np.ndarray) or not isinstance(vec_b, np.ndarray): return 0.0
    if vec_a.shape != vec_b.shape or vec_a.ndim != 1: return 0.0
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0: return 0.0
    return np.dot(vec_a, vec_b) / (norm_a * norm_b + DEFAULT_EPSILON)

def softmax_np(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if not isinstance(scores, np.ndarray) or scores.size == 0: return np.array([])
    temp = max(temperature, DEFAULT_EPSILON) # Ensure temperature is positive
    scores_stable = scores - np.max(scores)    # For numerical stability
    exps = np.exp(scores_stable / temp)
    sum_exps = np.sum(exps)
    if sum_exps == 0: return np.ones_like(scores) / max(1, scores.size) # Uniform if sum is zero
    return exps / sum_exps

def normalize_vector_np(vec: np.ndarray) -> np.ndarray:
    if not isinstance(vec, np.ndarray): return np.array([])
    norm = np.linalg.norm(vec)
    if norm == 0: return vec.copy() # Avoid division by zero, return original (zero) vector
    return vec / norm

# Finite difference Laplacian for 1D grid (used in brain substrate PDEs)
def finite_difference_laplacian_1d(field, dx):
    lap = np.zeros_like(field)
    # Apply periodic boundary conditions for the Laplacian
    lap[0] = (field[-1] - 2*field[0] + field[1]) / (dx**2)
    lap[1:-1] = (field[:-2] - 2*field[1:-1] + field[2:]) / (dx**2)
    lap[-1] = (field[-2] - 2*field[-1] + field[0]) / (dx**2)
    return lap

# Interpolate field value at a specific position on a grid (used for neurons)
def interpolate_field_value(field_array, position_on_grid, grid_coords):
    # Assuming grid_coords are sorted and represent region indices (0, 1, 2, ...)
    # Position on grid is the region index (an integer)
    # Direct lookup is sufficient if position_on_grid is an integer index
    if isinstance(position_on_grid, (int, np.integer)) and 0 <= position_on_grid < len(field_array):
        return field_array[position_on_grid]
    else:
        # If position is not an integer region index, perhaps interpolate?
        # For this coupled simulation, neuron position is the region index.
        # If position_on_grid is a float, find the nearest region index and use its field value.
        nearest_region_idx = int(round(position_on_grid))
        nearest_region_idx = np.clip(nearest_region_idx, 0, len(field_array) - 1)
        return field_array[nearest_region_idx]


# Smoothed Square Well Potential and Force Functions (from particle simulation)
def smooth_step(r, a, b):
    """Smooth step function that transitions from 0 to 1 between a and b."""
    if r < a:
        return 0.0
    elif r > b:
        return 1.0
    else:
        # Ensure b-a is not zero to avoid division by zero
        if abs(b - a) < DEFAULT_EPSILON: # Using DEFAULT_EPSILON or similar small number
            return 0.5 # Midpoint, or could be 0.0 or 1.0 depending on how you want to handle this degenerate case
        t = (r - a) / (b - a)
        t = np.clip(t, 0.0, 1.0) # Ensure t is within [0, 1] due to potential float issues
        return t * t * (3.0 - 2.0 * t)

def smooth_step_derivative(r, a, b):
    """Derivative of the smooth step function."""
    if r < a or r > b:
        return 0.0
    else:
        # Ensure b-a is not zero
        delta_ab = b - a
        if abs(delta_ab) < DEFAULT_EPSILON:
            return 0.0 # Derivative is undefined or very large, return 0 for stability

        t = (r - a) / delta_ab
        t = np.clip(t, 0.0, 1.0) # Ensure t is within [0, 1]
        return 6.0 * t * (1.0 - t) / delta_ab


def smoothed_square_well_potential(r, sigma, lambda_factor, epsilon, transition_width, large_repulsion_strength_scale=1e3):
    """Smoothed Square Well potential energy function."""
    well_inner_edge = sigma
    well_outer_edge = lambda_factor * sigma

    if well_outer_edge <= well_inner_edge + transition_width : # Check if well is too narrow for distinct transitions
        # Fallback to a simpler hard-sphere like repulsion + attraction for very narrow wells
        if r < sigma: return np.inf # Hard core
        elif r < lambda_factor * sigma : return -epsilon # Well depth
        else: return 0.0 # Outside well

    # Use small buffer for transitions
    a_outer = well_outer_edge - transition_width / 2.0
    b_outer = well_outer_edge + transition_width / 2.0
    a_inner = well_inner_edge - transition_width / 2.0
    b_inner = well_inner_edge + transition_width / 2.0

    # Ensure sensible ranges for transitions and prevent overlap of smoothing regions
    # This condition ensures the repulsive smoothing doesn't extend beyond the attractive one
    if b_inner >= a_outer:
        # Adjust edges to prevent overlap: make the inner transition sharper if necessary
        # This scenario suggests transition_width is too large for the gap between sigma and lambda_sigma
        # Reduce the effective transition regions or revert to a sharper well
        # For simplicity, if they overlap, we can prioritize the hard core aspect
        if r < sigma: return np.inf
        elif r >= sigma and r < lambda_factor * sigma: return -epsilon
        else: return 0.0


    Large_K = large_repulsion_strength_scale * epsilon

    # Contribution from the well attraction (-epsilon)
    U_outer_transition = -epsilon * (1.0 - smooth_step(r, a_outer, b_outer))

    # Contribution from the hard core repulsion (goes to large positive value)
    U_inner_transition = Large_K * (1.0 - smooth_step(r, a_inner, b_inner))

    potential = U_outer_transition + U_inner_transition

    if r <= DEFAULT_EPSILON: # Use DEFAULT_EPSILON for small r check
         potential = np.inf

    return potential


def smoothed_square_well_force(r, sigma, lambda_factor, epsilon, transition_width, large_repulsion_strength_scale=1e3):
    """Smoothed Square Well force function (derivative of smoothed potential)."""
    well_inner_edge = sigma
    well_outer_edge = lambda_factor * sigma

    if well_outer_edge <= well_inner_edge + transition_width: # Check if well is too narrow
        return large_repulsion_strength_scale * epsilon * 10 if r < sigma else 0.0 # Fallback strong repulsion

    a_outer = well_outer_edge - transition_width / 2.0
    b_outer = well_outer_edge + transition_width / 2.0
    a_inner = well_inner_edge - transition_width / 2.0
    b_inner = well_inner_edge + transition_width / 2.0

    if b_inner >= a_outer :
        # If smoothing regions overlap, prioritize hard repulsion for safety
        return large_repulsion_strength_scale * epsilon * 10 if r < sigma else 0.0

    Large_K = large_repulsion_strength_scale * epsilon

    dUdr = epsilon * smooth_step_derivative(r, a_outer, b_outer) - Large_K * smooth_step_derivative(r, a_inner, b_inner)
    force_magnitude = -dUdr # Force is negative gradient of potential

    # If deep inside the core region (r < a_inner), force should be highly repulsive
    if r < a_inner: # More robust check for core repulsion
        force_magnitude = Large_K / (transition_width + DEFAULT_EPSILON) * 6.0 * 0.25 # Approx max derivative val. scaled by K
        force_magnitude = max(force_magnitude, large_repulsion_strength_scale * epsilon * 10) # Ensure very repulsive


    return force_magnitude


# --- Data Structures (from both scripts, adapted) ---

# From Latent Ai (Matrix Mind)
@dataclass
class DynamicToken: # Represents a conceptual region in the unified model
    id: int
    text: str # e.g., "Region_0", "Region_1"
    latent_vector: np.ndarray # Intrinsic vector representing this region/token

    def __post_init__(self):
        # Ensure latent_vector is initialized with the correct dimension if not provided or invalid
        if not isinstance(self.latent_vector, np.ndarray) or self.latent_vector.shape != (default_matrix_mind_params.LATENT_DIM,):
             rnd_seed = int(stable_hash(self.text) * 1e9)
             local_random_state = np.random.RandomState(seed=rnd_seed)
             self.latent_vector = normalize_vector_np(local_random_state.uniform(-1, 1, default_matrix_mind_params.LATENT_DIM))
        elif np.all(self.latent_vector == 0): # Also re-initialize if it's all zeros
             rnd_seed = int(stable_hash(self.text) * 1e9)
             local_random_state = np.random.RandomState(seed=rnd_seed)
             self.latent_vector = normalize_vector_np(local_random_state.uniform(-1, 1, default_matrix_mind_params.LATENT_DIM))


    def __hash__(self): return hash(self.text)
    def __eq__(self, other):
        if isinstance(other, DynamicToken): return self.text == other.text
        return False

@dataclass
class SingularityMatrixEffect: # Represents a major shift/event in the MatrixMind state
    effect_type: str # e.g., "ROTATE_CONTEXT", "SCALE_DIMENSIONS", "ADD_NOISE"
    params: Dict[str, Any] = field(default_factory=dict)
    strength: float = 0.5
    # Could be triggered by significant, persistent patterns in the neural substrate

    def get_transformation_matrix(self, dim: int) -> Optional[np.ndarray]:
        if self.effect_type == "ROTATE_CONTEXT":
            angle = self.params.get("angle", np.pi / 4 * self.strength)
            c, s = np.cos(angle), np.sin(angle)
            rot_matrix = np.eye(dim)
            if dim >= 2: # Rotate first two dimensions as an example
                rot_matrix[0,0] = c; rot_matrix[0,1] = -s
                rot_matrix[1,0] = s; rot_matrix[1,1] = c
            return rot_matrix
        elif self.effect_type == "SCALE_DIMENSIONS":
            base_scales = self.params.get("scales", np.random.uniform(1 - self.strength*0.5, 1 + self.strength*0.5, dim))
            scale_factors = np.maximum(0.1, np.minimum(3.0, base_scales))
            return np.diag(scale_factors[:dim])
        return np.eye(dim) # Default to identity

class AIEvolvingMatrixMind:
    def __init__(self, matrix_mind_params: MatrixMindHyperParameters):
        self.matrix_mind_params = matrix_mind_params
        self.latent_dim = self.matrix_mind_params.LATENT_DIM
        self.tokens: List[DynamicToken] = []
        self.token_to_id_map: Dict[str, int] = {}
        self.M_context: np.ndarray = np.eye(self.latent_dim) * 0.1 # Initialize M_context

        num_conceptual_regions = default_params_unified['num_conceptual_regions'] # Get from unified params

        # Create tokens for each conceptual region
        for i in range(num_conceptual_regions):
            token_text = f"Region_{i}"
            initial_vec = np.zeros(self.latent_dim)
            if i < self.latent_dim: # Ensure i is within bounds for one-hot like encoding
                initial_vec[i] = 1.0 # Make it one-hot like for region identity
            else: # If num_conceptual_regions > latent_dim, handle gracefully (though ideally they match)
                initial_vec = normalize_vector_np(np.random.RandomState(seed=i).uniform(-1,1,self.latent_dim))

            new_id = len(self.tokens)
            # Pass the correctly dimensioned initial_vec to DynamicToken
            token = DynamicToken(id=new_id, text=token_text, latent_vector=normalize_vector_np(initial_vec.copy()))
            self.tokens.append(token)
            self.token_to_id_map[token_text] = new_id

        self.singularities: Dict[int, SingularityMatrixEffect] = {} # token_id (of a region) -> Effect

    def get_token_by_id(self, token_id: int) -> Optional[DynamicToken]:
        if 0 <= token_id < len(self.tokens): return self.tokens[token_id]
        return None

    def update_M_context_with_vector(self, chosen_token_vector: np.ndarray, eta: float):
        if chosen_token_vector.ndim == 1:
            v_col = chosen_token_vector.reshape(-1, 1)
            outer_product_v = v_col @ v_col.T
            self.M_context = (1 - eta) * self.M_context + eta * outer_product_v
            self.M_context = (self.M_context + self.M_context.T) / 2 # Ensure symmetry

    def update_M_context_with_coactivation(self, vec1: np.ndarray, vec2: np.ndarray, strength: float, eta: float):
        """Update M_context based on co-activation of two regions/tokens."""
        v1_col = vec1.reshape(-1,1)
        v2_col = vec2.reshape(-1,1)
        interaction_term = (v1_col @ v2_col.T + v2_col @ v1_col.T) / 2 # Symmetrized interaction
        self.M_context = (1 - eta * strength) * self.M_context + (eta * strength) * interaction_term
        self.M_context = (self.M_context + self.M_context.T) / 2 # Ensure symmetry

    def add_singularity(self, token_id: int, effect_type: str, strength: float = 0.5, effect_params: Optional[Dict] = None):
        token = self.get_token_by_id(token_id)
        if token:
            self.singularities[token_id] = SingularityMatrixEffect(effect_type, effect_params or {}, strength)
            print(f"[MatrixMind] Token '{token.text}' (ID: {token_id}) is now a singularity: type='{effect_type}', strength={strength:.2f}.")


# From brain_v1 (Neural Substrate)
class Neuron:
    _next_id = 0
    def __init__(self, position, E_local, S_local, M_local, assigned_region_id: int, neuron_model_params: Dict[str, Any]):
        self.id = Neuron._next_id
        Neuron._next_id += 1
        self.position = position # This position is the region index in the coupled sim
        self.assigned_region_id = assigned_region_id

        # Use neuron_model_params passed during initialization
        self.a = neuron_model_params['f_a'](E_local, S_local)
        self.b = neuron_model_params['f_b'](E_local, S_local)
        self.c = neuron_model_params['f_c'](E_local, S_local)
        self.d = neuron_model_params['f_d'](E_local, S_local)

        self.v = self.c + random.uniform(-5, 5) # Initialize v near c
        self.u = self.b * self.v
        self.spike_times = deque(maxlen=MAX_RECENT_SPIKES_PER_NEURON)

    def __repr__(self):
        return f"N(id={self.id}, r={self.assigned_region_id}, pos={self.position:.2f}, v={self.v:.1f})"


class UnifiedBrainSimulation:
    def __init__(self, brain_params: Dict[str, Any]):
        self.brain_params = brain_params
        self.matrix_mind_params = MatrixMindHyperParameters(**self.brain_params['matrix_mind'])
        self.matrix_mind = AIEvolvingMatrixMind(self.matrix_mind_params)
        print(f"Initialized MatrixMind with {len(self.matrix_mind.tokens)} regional tokens.")

        self.num_conceptual_regions = self.brain_params['unified']['num_conceptual_regions']
        self.x_grid = np.arange(self.num_conceptual_regions) # Grid points are region indices
        self.dx = 1.0 # Spacing between regions is 1.0

        # Substrate fields are now arrays over the regions
        self.E_field = np.full(self.num_conceptual_regions, 0.1) + 0.05 * np.sin(np.pi * self.x_grid / max(1, self.num_conceptual_regions - 1 if self.num_conceptual_regions > 0 else 1))
        self.S_field = np.full(self.num_conceptual_regions, self.brain_params['substrate_pde']['S_0'])
        self.M_field = np.zeros(self.num_conceptual_regions)
        self.A_field = np.zeros(self.num_conceptual_regions) # Activity field per region

        self.neurons: Dict[int, Neuron] = {}
        self.synapses: Dict[Tuple[int, int], Dict[str, Any]] = {} # { (pre_id, post_id) : {'weight': w} }
        self.synaptic_vars_s: Dict[int, float] = {} # { neuron_id: s_value }

        Neuron._next_id = 0 # Reset neuron ID counter for each simulation instance

        # Initialize neurons, assign them to regions
        for _ in range(self.brain_params['neurogenesis']['initial_N_neurons']):
            self._add_new_neuron_to_system(is_initial=True)

        self.t_current = 0.0
        self.outer_step_count = 0
        self.history = self._initialize_history()
        self.full_spike_history_raster = defaultdict(list)

        self.current_thought_sequence: List[int] = []
        self.current_thought_step_idx = 0
        self.thought_application_timer = 0
        # external_stimulus_Q is now per region
        self.external_stimulus_Q = np.zeros(self.num_conceptual_regions)
        self.current_adaptive_factors_per_brain_region = np.ones(self.num_conceptual_regions) # Added for MAUS input


    def _initialize_history(self):
        return {
            'time': [0.0],
            'N_neurons': [len(self.neurons)],
            'N_synapses': [len(self.synapses)],
            'avg_firing_rate': [0.0],
            'E_field': [self.E_field.copy()],
            'S_field': [self.S_field.copy()],
            'M_field': [self.M_field.copy()],
            'A_field': [self.A_field.copy()],
            'M_context_trace': [np.trace(self.matrix_mind.M_context)],
            'active_thought_region': [-1]
        }

    def _get_region_id_for_position(self, position: float) -> int:
        # This function is not used in the brain sim anymore, neuron position IS region ID
        # It's kept for conceptual clarity if needed elsewhere, but neuron.position is integer region ID.
        return int(round(position)) # Should be an integer region ID


    def _add_new_neuron_to_system(self, is_initial: bool = False):
        neurogenesis_params = self.brain_params['neurogenesis']
        neuron_model_params = self.brain_params['neuron_model']

        if self.num_conceptual_regions == 0:
             print("Warning: No conceptual regions defined for neuron placement.")
             return None


        if is_initial:
            # Assign to a random region initially
            assigned_region = random.choice(range(self.num_conceptual_regions))
            new_pos = assigned_region # Neuron position is the region index
        else:
            # Proliferation based on E*S across regions
            prob_density = np.maximum(0, self.E_field * self.S_field)
            sum_prob_density = np.sum(prob_density)
            if sum_prob_density < DEFAULT_EPSILON or len(self.x_grid) == 0:
                 pdf = np.ones_like(self.x_grid) / max(1,len(self.x_grid)) if len(self.x_grid) > 0 else []
                 if not pdf.any(): # Cannot assign if no regions
                      print("Cannot assign new neuron, no regions or PDF.")
                      return None

            else:
                 pdf = prob_density / sum_prob_density

            # Choose a region index based on probability density
            if len(self.x_grid) == 0 :
                 print("Cannot assign new neuron, x_grid is empty.")
                 return None

            assigned_region = np.random.choice(len(self.x_grid), p=pdf)
            new_pos = assigned_region # Neuron position is the region index


        E_loc = self.E_field[assigned_region]
        S_loc = self.S_field[assigned_region]
        M_loc = self.M_field[assigned_region]

        new_n = Neuron(position=new_pos, E_local=E_loc, S_local=S_loc, M_local=M_loc,
                       assigned_region_id=assigned_region, neuron_model_params=neuron_model_params)
        self.neurons[new_n.id] = new_n
        self.synaptic_vars_s[new_n.id] = 0.0 # Initialize synaptic var for new neuron
        return new_n

    def _solve_substrate_pdes_step(self, dt: float):
        # This method seems to be unused currently in run_simulation_step
        # It's kept here for completeness or if it's intended to be used later.
        pde_params = self.brain_params['substrate_pde']

        # Apply external stimulus to Q_E, Q_S, Q_M as defined in their parameters
        Q_E = pde_params['Q_E_baseline'] + self.external_stimulus_Q # Current thought stimulation applies here
        Q_S = pde_params['Q_S_baseline'] # No direct activity Q_S in current setup
        Q_M = pde_params['Q_M_baseline'] # No direct activity Q_M in current setup

        lap_E = finite_difference_laplacian_1d(self.E_field, self.dx)
        lap_S = finite_difference_laplacian_1d(self.S_field, self.dx)
        lap_M = finite_difference_laplacian_1d(self.M_field, self.dx)

        # PDE dynamics (Euler forward for simplicity)
        # dEdt = D_E * lap(E) - k_E*E + rho_E*A + Q_E
        dEdt = pde_params['D_E']*lap_E - pde_params['k_E']*self.E_field + pde_params['rho_E']*self.A_field + Q_E
        # dSdt = D_S * lap(S) - k_S*(S - S0) - gamma_S*A*S + Q_S
        dSdt = pde_params['D_S']*lap_S - pde_params['k_S']*(self.S_field - pde_params['S_0']) - pde_params['gamma_S']*self.A_field*self.S_field + Q_S
        # dMdt = D_M * lap(M) - k_M*M + rho_M*<A>_global + Q_M
        avg_A_global = np.mean(self.A_field) if self.A_field.size > 0 else 0
        dMdt = pde_params['D_M']*lap_M - pde_params['k_M']*self.M_field + pde_params['rho_M']*avg_A_global + Q_M


        self.E_field += dEdt * dt
        self.S_field += dSdt * dt
        self.M_field += dMdt * dt

        # Ensure fields remain non-negative
        self.E_field = np.maximum(0, self.E_field)
        self.S_field = np.maximum(0, self.S_field)
        self.M_field = np.maximum(0, self.M_field)


    def _run_neural_dynamics_and_plasticity_step(self, duration: float, current_time_offset: float):
        neuron_model_params = self.brain_params['neuron_model']
        synaptogenesis_params = self.brain_params['synaptogenesis']
        stdp_params = self.brain_params['stdp']
        time_params = self.brain_params['time']


        spikes_this_outer_step = defaultdict(list)
        accumulated_dW_for_stdp = defaultdict(float) # { (pre_id, post_id): accumulated_dw }

        neuron_ids_in_system = list(self.neurons.keys()) # Neurons active at start of this outer step
        if not neuron_ids_in_system: return spikes_this_outer_step, 0.0

        # Get field values at each neuron's region for this outer step
        # Assumes E, M, S fields are constant during the inner loop of one outer step
        field_values_at_neurons = {}
        for nid, n_obj in self.neurons.items():
            if 0 <= n_obj.assigned_region_id < self.num_conceptual_regions:
                 field_values_at_neurons[nid] = (
                     self.E_field[n_obj.assigned_region_id],
                     self.M_field[n_obj.assigned_region_id],
                     self.S_field[n_obj.assigned_region_id]
                 )
            else: # Fallback if region_id is somehow invalid (shouldn't happen)
                 field_values_at_neurons[nid] = (0,0,0)


        # External current from thought stimulation (per neuron, based on its region)
        external_current_per_neuron = {}
        stimulation_strength = self.brain_params['unified']['stimulation_strength_from_thought']
        for nid, n_obj in self.neurons.items():
            if 0 <= n_obj.assigned_region_id < self.num_conceptual_regions:
                 external_current_per_neuron[nid] = self.external_stimulus_Q[n_obj.assigned_region_id] * stimulation_strength
            else:
                 external_current_per_neuron[nid] = 0.0


        t_inner = 0
        while t_inner < duration:
            dt_actual_inner = min(time_params['DT_INNER_LOOP_NEURAL'], duration - t_inner)
            if dt_actual_inner < DEFAULT_EPSILON / 10: break # Avoid extremely small steps

            synaptic_currents_to_neurons = defaultdict(float)
            for (pre_id, post_id), syn_data in self.synapses.items():
                if pre_id in self.synaptic_vars_s and post_id in self.neurons: # Ensure both neurons exist
                    weight = syn_data['weight']
                    s_pre_val = self.synaptic_vars_s[pre_id]
                    synaptic_currents_to_neurons[post_id] += weight * s_pre_val

            current_abs_time_in_sim = current_time_offset + t_inner + dt_actual_inner
            spikes_this_dt_inner = [] # (neuron_id, spike_time)

            for nid in neuron_ids_in_system: # Iterate over neurons present at start of outer step
                if nid not in self.neurons: continue # Skip if neuron was deleted mid-outer-step (e.g., by neurogenesis logic if run more often)
                neuron = self.neurons[nid]

                I_syn = synaptic_currents_to_neurons[nid]
                E_loc, M_loc, S_loc = field_values_at_neurons.get(nid, (0,0,0)) # Use pre-calculated field values
                I_mod = neuron_model_params['eta_M_modulatory_current'] * M_loc
                I_stim_from_thought = external_current_per_neuron.get(nid, 0.0)
                total_input_current = I_syn + I_mod + I_stim_from_thought

                v, u = neuron.v, neuron.u
                # Izhikevich neuron model dynamics (using neuron-specific parameters a,b,c,d)
                dv_dt = (0.04*v**2 + 5*v + 140) - u + total_input_current
                du_dt = neuron.a * (neuron.b * v - u) # Use neuron's a, b

                neuron.v += dv_dt * dt_actual_inner
                neuron.u += du_dt * dt_actual_inner

                # Spike detection
                if neuron.v >= neuron_model_params['v_peak']:
                    neuron.v = neuron.c # Reset v using neuron's c
                    neuron.u += neuron.d # Increment u using neuron's d
                    neuron.spike_times.append(current_abs_time_in_sim)
                    spikes_this_dt_inner.append((nid, current_abs_time_in_sim))
                    spikes_this_outer_step[nid].append(current_abs_time_in_sim) # Accumulate for outer step

                    # Update synaptic variable 's' for this spiking neuron
                    self.synaptic_vars_s[nid] = self.synaptic_vars_s.get(nid, 0.0) + 1.0 # Increment s, ensure exists


            # Decay synaptic variables 's'
            tau_decay = synaptogenesis_params['tau_synaptic_variable_decay']
            if tau_decay > DEFAULT_EPSILON:
                decay_factor = math.exp(-dt_actual_inner / tau_decay)
                for n_id_s_var in list(self.synaptic_vars_s.keys()): # Iterate over copy if keys can change
                    self.synaptic_vars_s[n_id_s_var] *= decay_factor
                    self.synaptic_vars_s[n_id_s_var] = max(0.0, self.synaptic_vars_s[n_id_s_var]) # Ensure non-negative

            # STDP (Spike-Timing Dependent Plasticity)
            # Iterate through spikes that just occurred in this inner dt step
            for (spiking_neuron_id, t_spike) in spikes_this_dt_inner:
                if spiking_neuron_id not in self.neurons: continue
                E_spike, M_spike, S_spike = field_values_at_neurons.get(spiking_neuron_id, (0,0,0)) # Field values at spiking neuron's region

                # --- Post-synaptic spike (spiking_neuron_id is post) ---
                # Find pre-synaptic neurons that connect TO spiking_neuron_id
                for (pre_id, post_id), syn_data in self.synapses.items():
                    if post_id == spiking_neuron_id and pre_id in self.neurons:
                        for t_pre_spike in reversed(self.neurons[pre_id].spike_times): # Recent pre-spikes
                            if t_pre_spike >= t_spike : continue # Pre-spike must be BEFORE post-spike
                            delta_t = t_spike - t_pre_spike # delta_t > 0 for potentiation
                            if delta_t < stdp_params['max_dt_time_window']:
                                # Field values for modulation should be at the *post-synaptic* site (spiking_neuron_id)
                                dw = self._calculate_stdp_dw(delta_t, E_spike, M_spike, S_spike)
                                accumulated_dW_for_stdp[(pre_id, post_id)] += dw
                            else: break # Pre-spikes are too old

                # --- Pre-synaptic spike (spiking_neuron_id is pre) ---
                # Find post-synaptic neurons that spiking_neuron_id connects TO
                for (pre_id, post_id), syn_data in self.synapses.items():
                    if pre_id == spiking_neuron_id and post_id in self.neurons:
                        # Get field values at the *post-synaptic* neuron's region for depression modulation
                        E_post_target, M_post_target, S_post_target = field_values_at_neurons.get(post_id, (0,0,0))

                        for t_post_spike in reversed(self.neurons[post_id].spike_times): # Recent post-spikes of target
                            if t_post_spike >= t_spike: continue # Post-spike must be BEFORE pre-spike for this path
                            delta_t = t_post_spike - t_spike # delta_t < 0 for depression
                            if abs(delta_t) < stdp_params['max_dt_time_window']:
                                dw = self._calculate_stdp_dw(delta_t, E_post_target, M_post_target, S_post_target)
                                accumulated_dW_for_stdp[(pre_id, post_id)] += dw
                            else: break # Post-spikes are too old

            t_inner += dt_actual_inner


        # Apply accumulated weight changes after the inner loop finishes
        synapses_to_prune = []
        for (pre_id, post_id), total_dw in accumulated_dW_for_stdp.items():
            if (pre_id, post_id) in self.synapses:
                self.synapses[(pre_id, post_id)]['weight'] += total_dw
                # Clip weights to be within [w_min, w_max]
                self.synapses[(pre_id, post_id)]['weight'] = np.clip(
                    self.synapses[(pre_id, post_id)]['weight'],
                    stdp_params['w_min_synaptic_weight'], stdp_params['w_max_synaptic_weight']
                )
                # Prune synapses if weight drops below minimum (or very close to it)
                if self.synapses[(pre_id, post_id)]['weight'] < stdp_params['w_min_synaptic_weight'] + DEFAULT_EPSILON:
                    synapses_to_prune.append((pre_id, post_id))

        # Remove pruned synapses
        for syn_key in synapses_to_prune:
            if syn_key in self.synapses:
                 del self.synapses[syn_key]


        # Calculate average firing rate for this outer step
        total_spikes_in_outer_step = sum(len(s_list) for s_list in spikes_this_outer_step.values())
        # Use number of neurons at the start of this outer step for rate calculation
        num_neurons_for_rate = len(neuron_ids_in_system)
        avg_rate = total_spikes_in_outer_step / max(1, num_neurons_for_rate) / duration if duration > 0 else 0.0

        return spikes_this_outer_step, avg_rate


    def _calculate_stdp_dw(self, delta_t: float, E_post: float, M_post: float, S_post: float) -> float:
        stdp_params = self.brain_params['stdp']
        if delta_t > 0: # Potentiation (post-synaptic spike after pre-synaptic)
            # Ensure tau_plus_potentiation is positive to avoid division by zero or math errors
            tau_plus = stdp_params['tau_plus_potentiation']
            if tau_plus <= DEFAULT_EPSILON: return 0.0
            factor = math.exp(-delta_t / tau_plus)
            A_plus = stdp_params['f_Aplus_substrate_modulation'](E_post, M_post, S_post)
            return A_plus * factor
        elif delta_t < 0: # Depression (pre-synaptic spike after post-synaptic)
            # Ensure tau_minus_depression is positive
            tau_minus = stdp_params['tau_minus_depression']
            if tau_minus <= DEFAULT_EPSILON: return 0.0
            factor = math.exp(delta_t / tau_minus) # delta_t is negative
            A_minus = stdp_params['f_Aminus_substrate_modulation'](E_post, M_post, S_post)
            return -A_minus * factor # Depression is a negative weight change
        return 0.0


    def _perform_structural_plasticity_step(self, dt: float, adaptive_factors_per_brain_region: np.ndarray):
        neurogenesis_params = self.brain_params['neurogenesis']
        synaptogenesis_params = self.brain_params['synaptogenesis']

        # --- Neurogenesis (Neuron Creation) ---
        current_n_neurons = len(self.neurons)
        if current_n_neurons < neurogenesis_params['N_max_total_neurons']:
            # Proliferation potential based on E*S integral over regions
            if self.num_conceptual_regions > 0 and self.E_field.size == self.num_conceptual_regions and self.S_field.size == self.num_conceptual_regions:
                 integral_ES = np.sum(np.maximum(0, self.E_field * self.S_field)) * self.dx # Sum over regions
                 proliferation_potential = neurogenesis_params['alpha_proliferation_rate'] * integral_ES * \
                                          max(0, (1 - current_n_neurons / neurogenesis_params['N_max_total_neurons']))

                # Apply adaptive factor to proliferation potential based on brain region
                 if adaptive_factors_per_brain_region is not None and adaptive_factors_per_brain_region.shape[0] == self.num_conceptual_regions:
                    # Scale based on the specific region, averaged here as brain acts more holistically than particles in their region
                    avg_adaptive_factor = np.mean(adaptive_factors_per_brain_region)
                    proliferation_potential *= avg_adaptive_factor


                # Determine number of new neurons to add based on potential and dt
                 num_new_neurons_expected = proliferation_potential * dt
                 num_new_neurons_actual = np.random.poisson(num_new_neurons_expected)

                 for _ in range(num_new_neurons_actual):
                     if len(self.neurons) >= neurogenesis_params['N_max_total_neurons']: break # Stop if max reached
                     self._add_new_neuron_to_system() # Adds to a region based on E*S
            # else:
                 # print("Skipping neurogenesis due to field size mismatch or no regions.")


        # --- Synaptogenesis (Synapse Formation) ---
        new_syn_count = 0
        neuron_ids_list = list(self.neurons.keys())
        if len(neuron_ids_list) < 2: return # Need at least 2 neurons for a synapse

        # Randomly sample pairs of neurons to check for potential synapse formation
        # Limit the number of pairs checked per step for performance
        num_pairs_to_check = min(len(neuron_ids_list)**2, max(100, int(synaptogenesis_params['prob_formation_scale'] * len(neuron_ids_list)**2 * 10))) # Heuristic limit
        # Ensure num_pairs_to_check is at least 0
        num_pairs_to_check = max(0, int(num_pairs_to_check))


        for _ in range(num_pairs_to_check):
            if len(neuron_ids_list) < 2: break
            try:
                idx_pre, idx_post = random.sample(range(len(neuron_ids_list)), 2)
            except ValueError: break
            pre_id, post_id = neuron_ids_list[idx_pre], neuron_ids_list[idx_post]

            # Skip if synapse already exists
            if (pre_id, post_id) in self.synapses: continue

            # Ensure neurons still exist (might be deleted by other processes if run concurrently, though not in this structure)
            if pre_id not in self.neurons or post_id not in self.neurons: continue
            pre_n, post_n = self.neurons[pre_id], self.neurons[post_id]

            # Check distance in terms of region index difference
            region_dist = abs(pre_n.assigned_region_id - post_n.assigned_region_id)
            if region_dist > synaptogenesis_params['max_connection_region_dist']: continue

            # Get substrate field values at the midpoint region (or average of regions)
            if not (0 <= pre_n.assigned_region_id < self.num_conceptual_regions and \
                    0 <= post_n.assigned_region_id < self.num_conceptual_regions) :
                continue # Skip if region IDs are invalid

            avg_E_mid = (self.E_field[pre_n.assigned_region_id] + self.E_field[post_n.assigned_region_id]) / 2.0
            avg_S_mid = (self.S_field[pre_n.assigned_region_id] + self.S_field[post_n.assigned_region_id]) / 2.0

            # Check S threshold
            if avg_S_mid < synaptogenesis_params['S_threshold_for_connection']: continue

            # Probability of formation based on distance and substrate fields
            # Distance factor: decays with region distance
            max_conn_dist_scaled = synaptogenesis_params['max_connection_region_dist'] / 3.0
            dist_factor = math.exp(-0.5 * (region_dist / max(max_conn_dist_scaled, DEFAULT_EPSILON))**2) # Gaussian decay, avoid div by zero

            # Substrate factor: increases with E
            substrate_factor = 1 + synaptogenesis_params['E_factor_on_connection_prob'] * avg_E_mid
            # Total probability of formation in this time step
            prob_form_synapse = synaptogenesis_params['prob_formation_scale'] * dist_factor * substrate_factor * dt

            # Form synapse with this probability
            if random.random() < prob_form_synapse:
                # Initialize synapse weight
                weight = max(DEFAULT_EPSILON, random.gauss(
                    synaptogenesis_params['initial_weight_mean'],
                    synaptogenesis_params['initial_weight_std']
                ))
                weight = min(weight, default_params_stdp['w_max_synaptic_weight']) # Ensure initial weight is not > max

                self.synapses[(pre_id, post_id)] = {'weight': weight}
                new_syn_count +=1
        # print(f"Step {self.outer_step_count}: Added {new_syn_count} new synapses.")


    def _update_activity_field_A(self, spikes_in_step: Dict[int, List[float]], dt: float):
        activity_field_params = self.brain_params['activity_field']
        if self.num_conceptual_regions == 0: return # No regions to update

        # Accumulate spike counts per region in this outer step
        regional_spike_counts = np.zeros(self.num_conceptual_regions)
        for nid, spike_times in spikes_in_step.items():
            if nid in self.neurons and spike_times:
                neuron_obj = self.neurons[nid]
                if 0 <= neuron_obj.assigned_region_id < self.num_conceptual_regions:
                     regional_spike_counts[neuron_obj.assigned_region_id] += len(spike_times)

        # Smooth the regional spike counts spatially
        sigma_regions = activity_field_params['kernel_sigma_spatial_smoothing_regions']
        if sigma_regions > DEFAULT_EPSILON and self.num_conceptual_regions > 1 : # Smoothing requires multiple regions
            # Apply Gaussian smoothing across the regional indices (using discrete grid points)
            smoothed_regional_activity = gaussian_filter1d(regional_spike_counts, sigma=sigma_regions, mode='wrap') # Use wrap for periodic
        else:
            smoothed_regional_activity = regional_spike_counts

        # Update the activity field A with temporal decay
        tau_A = activity_field_params['tau_A_temporal_decay']
        if tau_A > DEFAULT_EPSILON:
             decay_factor = math.exp(-dt / tau_A)
             self.A_field = self.A_field * decay_factor + smoothed_regional_activity * (1-decay_factor) # More conventional update
        else:
             self.A_field = smoothed_regional_activity # No temporal decay, just replace

        # Ensure activity field remains non-negative
        self.A_field = np.maximum(0, self.A_field)


    def _update_matrix_mind_from_neural_state(self):
        unified_params = self.brain_params['unified']
        matrix_mind_params_dict = self.brain_params['matrix_mind'] # This is a dict
        num_regions = self.num_conceptual_regions
        if num_regions == 0: return


        # --- 1. Activity-based update (Functional Co-activation) ---
        # Use the current A_field (regional activity) to update M_context
        region_activities = self.A_field.copy()

        sum_region_activities = np.sum(region_activities)
        if sum_region_activities > DEFAULT_EPSILON:
            normalized_region_activities = region_activities / sum_region_activities
        else:
            normalized_region_activities = np.zeros_like(region_activities)

        eta_activity = matrix_mind_params_dict['CONTEXT_MATRIX_ETA'] # Access as dict
        for i in range(num_regions):
            for j in range(i, num_regions): # Iterate to include self-coactivation (i==j)
                # Co-activation strength between region i and j
                co_activation_strength = normalized_region_activities[i] * normalized_region_activities[j]

                if co_activation_strength > 1e-6: # Threshold for significant co-activation
                    # Get the latent vectors for the tokens corresponding to these regions
                    token_i = self.matrix_mind.get_token_by_id(i)
                    token_j = self.matrix_mind.get_token_by_id(j)

                    if token_i and token_j:
                         self.matrix_mind.update_M_context_with_coactivation(
                             token_i.latent_vector, token_j.latent_vector,
                             strength=co_activation_strength,
                             eta=eta_activity
                         )
        M_context_after_activity_update = self.matrix_mind.M_context.copy()


        # --- 2. Structural Connectivity-based update ---
        structural_connectivity_matrix = np.zeros((num_regions, num_regions))
        for (pre_nid, post_nid), syn_data in self.synapses.items():
            if pre_nid in self.neurons and post_nid in self.neurons:
                neuron_pre = self.neurons[pre_nid]
                neuron_post = self.neurons[post_nid]
                if 0 <= neuron_pre.assigned_region_id < num_regions and 0 <= neuron_post.assigned_region_id < num_regions:
                    r_pre = neuron_pre.assigned_region_id
                    r_post = neuron_post.assigned_region_id
                    structural_connectivity_matrix[r_pre, r_post] += syn_data['weight'] # Sum weights between regions

        # Normalize structural connectivity (e.g., by max observed weight or sum)
        max_observed_conn = np.max(structural_connectivity_matrix)
        if max_observed_conn > DEFAULT_EPSILON:
            norm_structural_connectivity = structural_connectivity_matrix / max_observed_conn
        else:
            norm_structural_connectivity = np.zeros_like(structural_connectivity_matrix)

        # Construct the target M_context based on structure
        M_target_from_structure = np.zeros_like(self.matrix_mind.M_context)
        for r1 in range(num_regions):
            for r2 in range(num_regions): # Iterate all pairs for outer products
                conn_strength_r1_r2 = (norm_structural_connectivity[r1, r2] + norm_structural_connectivity[r2, r1]) / 2.0

                if conn_strength_r1_r2 > DEFAULT_EPSILON:
                    token_r1 = self.matrix_mind.get_token_by_id(r1)
                    token_r2 = self.matrix_mind.get_token_by_id(r2)

                    if token_r1 and token_r2:
                         interaction_op = (token_r1.latent_vector.reshape(-1, 1) @ token_r2.latent_vector.reshape(1, -1) + \
                                           token_r2.latent_vector.reshape(-1, 1) @ token_r1.latent_vector.reshape(1, -1)) / 2.0
                         M_target_from_structure += conn_strength_r1_r2 * interaction_op

        # Normalize M_target_from_structure if its scale is too different
        norm_M_target = np.linalg.norm(M_target_from_structure)
        if norm_M_target > DEFAULT_EPSILON:
             norm_M_activity = np.linalg.norm(M_context_after_activity_update)
             if norm_M_activity > DEFAULT_EPSILON: # Avoid scaling by zero
                  M_target_from_structure = M_target_from_structure * (norm_M_activity / norm_M_activity) # Scale M_target to M_activity norm if exists

        # Blend the activity-updated M_context with the structural target M_context
        eta_struct = unified_params['structural_influence_eta']
        self.matrix_mind.M_context = (1 - eta_struct) * M_context_after_activity_update + \
                                     eta_struct * M_target_from_structure

        # --- 3. Final Decay/Identity Mixing for M_context Stability ---
        mean_diag = np.mean(np.diag(self.matrix_mind.M_context)) if self.matrix_mind.M_context.size > 0 and self.matrix_mind.M_context.ndim == 2 else 0.1
        self.matrix_mind.M_context = 0.995 * self.matrix_mind.M_context + \
                                     0.005 * np.eye(self.matrix_mind.latent_dim) * mean_diag
        self.matrix_mind.M_context = (self.M_context + self.M_context.T) / 2.0 # Ensure symmetry


    def _generate_and_apply_matrix_mind_thought(self):
        unified_params = self.brain_params['unified']
        matrix_mind_params_dict = self.brain_params['matrix_mind'] # dict
        if self.num_conceptual_regions == 0:
            self.current_thought_sequence = []
            self.thought_application_timer = 0
            return

        # Start thought sequence from the region with highest activity
        region_activities_for_start = self.A_field.copy()

        if np.any(region_activities_for_start > DEFAULT_EPSILON) and region_activities_for_start.size > 0 :
            start_region_id_for_thought = np.argmax(region_activities_for_start)
        else:
            # If no activity, start from a random region if regions exist
            start_region_id_for_thought = random.choice(range(self.num_conceptual_regions)) if self.num_conceptual_regions > 0 else -1

        if start_region_id_for_thought == -1 : # No valid start region
            self.current_thought_sequence = []
            self.thought_application_timer = 0
            return


        # Get the token ID corresponding to the starting region ID
        start_token_id = start_region_id_for_thought # Token ID = Region ID in this system

        # Intention vector can be added here if needed (e.g., from external input or goals)
        intention_vec_np = None # Placeholder

        # Generate the thought sequence (list of token IDs)
        path_token_ids, _ = self._matrix_mind_generate_thought_sequence(
            start_token_id=start_token_id,
            intention_vector_np=intention_vec_np
        )

        if path_token_ids:
            self.current_thought_sequence = path_token_ids
            self.current_thought_step_idx = 0 # Start applying from the first token in sequence
            self.thought_application_timer = unified_params['thought_application_duration'] # Duration to apply stimulation
            # print(f"[MatrixMind] Generated sequence: {[self.matrix_mind.tokens[tid].text for tid in path_token_ids if tid < len(self.matrix_mind.tokens)]}")
        else:
            self.current_thought_sequence = []
            self.thought_application_timer = 0
            # print(f"[MatrixMind] No thought sequence generated.")

        # Apply the first step of the thought sequence to the neural substrate
        self._apply_current_thought_to_neural_substrate()


    def _apply_current_thought_to_neural_substrate(self):
        unified_params = self.brain_params['unified']

        # Reset external stimulation for all regions
        self.external_stimulus_Q.fill(0.0)
        active_thought_region_for_history = -1 # Default to no active region

        # If there is an active thought sequence and the timer is running
        if self.current_thought_sequence and self.thought_application_timer > 0:
            # Check if the current step index is within the sequence length
            if self.current_thought_step_idx < len(self.current_thought_sequence):
                # Get the token ID for the current thought step
                active_region_token_id = self.current_thought_sequence[self.current_thought_step_idx]

                # The token ID corresponds directly to the region ID in this coupled system
                active_region_id = active_region_token_id

                # Apply stimulation to the neural substrate field (Q_E) in the targeted region
                if 0 <= active_region_id < self.num_conceptual_regions:
                    self.external_stimulus_Q[active_region_id] = 1.0 # Apply a unit stimulation (will be scaled in brain dynamics)
                    active_thought_region_for_history = active_region_id

                # Move to the next step in the thought sequence for the next outer simulation step
                self.current_thought_step_idx += 1
            else: # Reached end of sequence, but timer might still be on
                 pass # No more steps to apply, stimulation remains off or as per last applied step


            # Decrease the thought application timer
            self.thought_application_timer -=1

            # If the timer runs out or sequence ends, clear the thought sequence and reset stimulation implicitly next call
            if self.thought_application_timer <= 0 or self.current_thought_step_idx >= len(self.current_thought_sequence):
                self.current_thought_sequence = []
                self.current_thought_step_idx = 0 # Reset index for next sequence
                self.thought_application_timer = 0 # Ensure timer is zero
                # Stimulation (external_stimulus_Q) is already reset to 0.0 at the start of this function
                # active_thought_region_for_history will remain -1 if sequence cleared here


        # Update history for active thought region
        # Ensure history list for 'active_thought_region' is correctly managed
        if len(self.history['active_thought_region']) <= self.outer_step_count: # if new step for history
             self.history['active_thought_region'].append(active_thought_region_for_history)
        else: # if overwriting/updating existing step's history entry
             self.history['active_thought_region'][self.outer_step_count] = active_thought_region_for_history



    def _matrix_mind_generate_thought_sequence(self, start_token_id: int, intention_vector_np: Optional[np.ndarray]):
        matrix_mind_params_dict = self.brain_params['matrix_mind'] # dict
        path_ids: List[int] = []
        if not self.matrix_mind.tokens: return path_ids, []

        current_token_id = start_token_id
        if not (0 <= current_token_id < len(self.matrix_mind.tokens)):
            # If invalid start_token_id, try to pick a random one if possible
            current_token_id = random.choice(range(len(self.matrix_mind.tokens))) if self.matrix_mind.tokens else -1
            if current_token_id == -1: return path_ids, [] # Cannot proceed


        previous_chosen_eigenvector: Optional[np.ndarray] = None
        current_M_context_for_thought = self.matrix_mind.M_context.copy()

        for step_num_in_seq in range(matrix_mind_params_dict['N_STEPS_PER_THOUGHT_SEQUENCE']):
            path_ids.append(current_token_id)
            operational_M = current_M_context_for_thought
            M_symmetric = (operational_M + operational_M.T) / 2
            try:
                # Calculate eigenvalues and eigenvectors
                eigenvalues, eigenvectors_cols = np.linalg.eigh(M_symmetric)
                eigenvectors_rows = eigenvectors_cols.T # Eigenvectors as rows
            except np.linalg.LinAlgError:
                # Fallback: choose next token randomly (excluding current)
                if len(self.matrix_mind.tokens) > 1:
                    choices = [i for i in range(len(self.matrix_mind.tokens)) if i != current_token_id]
                    next_token_id = random.choice(choices) if choices else current_token_id
                elif len(self.matrix_mind.tokens) == 1:
                     next_token_id = current_token_id # Stay if only one token
                else: break # No tokens, should not happen if checked before
                current_token_id = next_token_id
                if step_num_in_seq < matrix_mind_params_dict['N_STEPS_PER_THOUGHT_SEQUENCE'] -1 : # If not the last step for path_ids
                     continue
                else: # If it's the last iteration, path_ids has already been appended
                     break


            # Filter out eigenvectors with very small eigenvalues
            valid_eigen_indices = [i for i, val in enumerate(eigenvalues) if abs(val) > matrix_mind_params_dict['MIN_EIGENVALUE_THRESHOLD']] # Check abs for stability
            if not valid_eigen_indices: break

            # Score valid eigenvectors
            eigenvector_scores = np.zeros(len(valid_eigen_indices))
            # Get the latent vector for the current token (region)
            current_token_vec_for_coherence = self.matrix_mind.tokens[current_token_id].latent_vector

            for i, eig_idx in enumerate(valid_eigen_indices):
                ev = eigenvectors_rows[eig_idx]
                score = abs(eigenvalues[eig_idx]) # Base score is eigenvalue magnitude

                # Novelty bias
                if previous_chosen_eigenvector is not None:
                    novelty = (1 - abs(cosine_similarity_np(ev, previous_chosen_eigenvector)))
                    score += matrix_mind_params_dict['NOVELTY_BIAS_STRENGTH'] * novelty * abs(eigenvalues[eig_idx])

                # Coherence bias with current token
                coherence = abs(cosine_similarity_np(ev, current_token_vec_for_coherence))
                score += matrix_mind_params_dict['EIGENVECTOR_COHERENCE_STRENGTH'] * coherence * abs(eigenvalues[eig_idx])


                # Intention bias
                if intention_vector_np is not None and intention_vector_np.size == self.matrix_mind.latent_dim : # Ensure valid intention vector
                    alignment = abs(cosine_similarity_np(ev, intention_vector_np))
                    score += matrix_mind_params_dict['INTENTION_STRENGTH'] * alignment * abs(eigenvalues[eig_idx])
                eigenvector_scores[i] = score


            # Select the top N eigenvectors based on scores
            num_to_select_ev = min(matrix_mind_params_dict['NUM_EIGENVECTORS_TO_CONSIDER'], len(valid_eigen_indices))
            top_scored_indices_in_valid = np.argsort(eigenvector_scores)[-num_to_select_ev:]
            target_eigenvectors_for_projection = [eigenvectors_rows[valid_eigen_indices[j]] for j in top_scored_indices_in_valid]

            if not target_eigenvectors_for_projection: break

            # Score candidate tokens (regions)
            candidate_token_scores_for_next_step = np.zeros(len(self.matrix_mind.tokens))
            for token_idx_cand, token_cand in enumerate(self.matrix_mind.tokens):
                token_vec_cand = token_cand.latent_vector
                total_alignment = sum(abs(cosine_similarity_np(token_vec_cand, target_ev)) for target_ev in target_eigenvectors_for_projection)
                candidate_token_scores_for_next_step[token_idx_cand] = total_alignment / len(target_eigenvectors_for_projection) if target_eigenvectors_for_projection else 0

            # Penalize staying
            if len(self.matrix_mind.tokens) > 1:
                 candidate_token_scores_for_next_step[current_token_id] *= 0.5

            # Select next token
            token_probs = softmax_np(candidate_token_scores_for_next_step, matrix_mind_params_dict['TEMPERATURE'])
            if token_probs.size == 0 or np.any(np.isnan(token_probs)) or np.sum(token_probs) < DEFAULT_EPSILON:
                choices = [i for i in range(len(self.matrix_mind.tokens)) if i != current_token_id] or list(range(len(self.matrix_mind.tokens)))
                next_token_id = random.choice(choices) if choices else current_token_id
            else:
                next_token_id = np.random.choice(len(self.matrix_mind.tokens), p=token_probs)


            # Update M_context for thought process
            chosen_token_for_next_step_vec = self.matrix_mind.tokens[next_token_id].latent_vector
            v_col_thought = chosen_token_for_next_step_vec.reshape(-1,1)
            # Smaller learning rate for thought internal M_context update
            eta_thought_internal = matrix_mind_params_dict['CONTEXT_MATRIX_ETA'] * 0.1
            current_M_context_for_thought = (1 - eta_thought_internal) * current_M_context_for_thought + \
                                            eta_thought_internal * (v_col_thought @ v_col_thought.T)
            current_M_context_for_thought = (current_M_context_for_thought + current_M_context_for_thought.T)/2


            # Update previously chosen eigenvector
            if target_eigenvectors_for_projection:
                 alignments_to_chosen_token = [cosine_similarity_np(chosen_token_for_next_step_vec, ev) for ev in target_eigenvectors_for_projection]
                 if alignments_to_chosen_token:
                    previous_chosen_eigenvector = target_eigenvectors_for_projection[np.argmax(alignments_to_chosen_token)]


            current_token_id = next_token_id

            # Random jump in thought
            if random.random() < matrix_mind_params_dict['RANDOM_JUMP_PROBABILITY'] * 0.1:
                rand_perturb_vec = normalize_vector_np(np.random.rand(self.matrix_mind.latent_dim) - 0.5)
                current_M_context_for_thought += 0.01 * (rand_perturb_vec.reshape(-1,1) @ rand_perturb_vec.reshape(1,-1))
                current_M_context_for_thought = (current_M_context_for_thought + current_M_context_for_thought.T)/2

        return path_ids, []


    def run_simulation_step(self, dt: float, particle_density_per_region: np.ndarray):
        """
        Runs one outer time step of the Unified Brain Simulation.
        Receives particle density from the coupled simulation.
        """
        unified_params = self.brain_params['unified']
        time_params = self.brain_params['time']

        # --- Input from Particle Simulation: Update Brain Activity Field ---
        if particle_density_per_region.shape[0] == self.num_conceptual_regions and self.num_conceptual_regions > 0 :
             # Activity from particle density directly adds to A_field (scaled)
             # This could be done before or after A_field decay, or as part of smoothed_regional_activity
             # Let's consider it an input source for A_field before the final decay/update
             self.A_field += particle_density_per_region * unified_params['particle_density_to_brain_activity_scale']
             self.A_field = np.maximum(0, self.A_field)

        # --- Solve Substrate PDE Step (E,S,M fields) ---
        self._solve_substrate_pdes_step(dt) # Update E, S, M fields based on A and Q

        # --- Run Neural Dynamics and Plasticity (Inner Loop) ---
        spikes_this_step, avg_firing_rate = self._run_neural_dynamics_and_plasticity_step(
            duration=dt, current_time_offset=self.t_current
        )
        for nid, s_times in spikes_this_step.items():
            self.full_spike_history_raster[nid].extend(s_times)

        # --- Perform Structural Plasticity (Neurogenesis and Synaptogenesis) ---
        # Pass zone-specific adaptive factor to modulate proliferation rate
        self._perform_structural_plasticity_step(dt, self.current_adaptive_factors_per_brain_region)

        # --- Update Brain Activity Field 'A' based on spikes this step ---
        self._update_activity_field_A(spikes_this_step, dt)
        # Note: The influence from particle_density_per_region is already added to A_field.
        # _update_activity_field_A will then process it (smoothing, decay).


        # --- Update Matrix Mind and Generate/Apply Thought Sequence ---
        if self.outer_step_count % unified_params['matrix_mind_update_interval'] == 0:
            self._update_matrix_mind_from_neural_state()
            self._generate_and_apply_matrix_mind_thought() # Generates new and applies first step
        else:
            self._apply_current_thought_to_neural_substrate() # Continues applying or finishes current thought

        # --- Output to Particle Simulation: brain_stimulation_per_region ---
        # self.external_stimulus_Q holds the stimulation pattern from the thought application.
        # This will be read by the CoupledSimulation to apply forces.
        brain_stimulation_per_region = self.external_stimulus_Q.copy() # Return a copy


        # Increment brain simulation time and outer step count
        self.t_current += dt

        # --- History Recording ---
        # Current_history_len might be 1 if it's the first step
        current_history_len = len(self.history['time'])
        if self.outer_step_count >= current_history_len : # If this is a new step to record
            self.history['time'].append(self.t_current)
            self.history['N_neurons'].append(len(self.neurons))
            self.history['N_synapses'].append(len(self.synapses))
            self.history['avg_firing_rate'].append(avg_firing_rate)
            self.history['E_field'].append(self.E_field.copy())
            self.history['S_field'].append(self.S_field.copy())
            self.history['M_field'].append(self.M_field.copy())
            self.history['A_field'].append(self.A_field.copy())
            self.history['M_context_trace'].append(np.trace(self.matrix_mind.M_context))
            # active_thought_region is appended in _apply_current_thought_to_neural_substrate
            # Need to ensure it matches the current step index (self.outer_step_count)
            if len(self.history['active_thought_region']) <= self.outer_step_count:
                self.history['active_thought_region'].append(-1) # Append default if not set by apply_thought

        else: # Update existing history entry (e.g., if called multiple times for same outer_step_count)
            self.history['time'][self.outer_step_count] = self.t_current
            self.history['N_neurons'][self.outer_step_count] = len(self.neurons)
            self.history['N_synapses'][self.outer_step_count] = len(self.synapses)
            self.history['avg_firing_rate'][self.outer_step_count] = avg_firing_rate
            self.history['E_field'][self.outer_step_count] = self.E_field.copy()
            self.history['S_field'][self.outer_step_count] = self.S_field.copy()
            self.history['M_field'][self.outer_step_count] = self.M_field.copy()
            self.history['A_field'][self.outer_step_count] = self.A_field.copy()
            self.history['M_context_trace'][self.outer_step_count] = np.trace(self.matrix_mind.M_context)
            # active_thought_region updated in _apply_current_thought_to_neural_substrate


        self.outer_step_count += 1 # Increment after processing and recording for current step
        return brain_stimulation_per_region


    def run_full_simulation(self):
        """
        Runs the Unified Brain Simulation independently for testing/debugging.
        In the coupled simulation, this is replaced by the CoupledSimulation loop.
        """
        print("--- Starting Independent Unified Brain Simulation ---")
        sim_start_time = time.time()
        dummy_particle_density = np.zeros(self.num_conceptual_regions)

        total_steps_brain = int(self.brain_params['time']['T_FINAL_SIMULATION'] / self.brain_params['time']['DT_OUTER_LOOP'])

        for step_idx in range(total_steps_brain): # Use outer_step_count for loop condition
            if self.t_current >= self.brain_params['time']['T_FINAL_SIMULATION']: break

            # Dummy adaptive factors for independent run
            self.current_adaptive_factors_per_brain_region = np.ones(self.num_conceptual_regions)

            self.run_simulation_step(self.brain_params['time']['DT_OUTER_LOOP'], dummy_particle_density)

            if (self.outer_step_count-1) % self.brain_params["time"]["PROGRESS_PRINT_INTERVAL"] == 0 or \
               self.t_current >= self.brain_params['time']['T_FINAL_SIMULATION']:

                # History index should be outer_step_count - 1 as it's incremented after recording
                hist_idx_to_print = self.outer_step_count -1
                if hist_idx_to_print < 0 : hist_idx_to_print = 0 # Ensure valid index

                if hist_idx_to_print < len(self.history['time']): # Check if history entry exists
                    print(f"Brain Step {self.outer_step_count-1}: t={self.history['time'][hist_idx_to_print]:.2f}s, "
                          f"N={self.history['N_neurons'][hist_idx_to_print]}, Syn={self.history['N_synapses'][hist_idx_to_print]}, "
                          f"Rate={self.history['avg_firing_rate'][hist_idx_to_print]:.2f}Hz, "
                          f"M_Trace={self.history['M_context_trace'][hist_idx_to_print]:.3f}, "
                          f"ThoughtRegion={self.history['active_thought_region'][hist_idx_to_print]}")
            if not self.neurons:
                print("No neurons remaining. Halting brain simulation.")
                break

        sim_end_time = time.time()
        print(f"--- Independent Brain Simulation Finished in {sim_end_time - sim_start_time:.2f} seconds ---")
        self.plot_results()


    def plot_results(self):
        """
        Generates plots for the Unified Brain Simulation history.
        """
        print("Generating brain plots...")
        # Ensure all history lists are truncated to the minimum common length
        min_len = 0
        if self.history and all(isinstance(v, list) for v in self.history.values()) and self.history.values():
             min_len = min(len(v) for v in self.history.values() if isinstance(v, list))


        if min_len == 0:
             print("No brain history data to plot.")
             return

        times_np = np.array(self.history['time'][:min_len])
        history_N_neurons = np.array(self.history['N_neurons'][:min_len])
        history_N_synapses = np.array(self.history['N_synapses'][:min_len])
        history_avg_firing_rate = np.array(self.history['avg_firing_rate'][:min_len])
        history_M_context_trace = np.array(self.history['M_context_trace'][:min_len])
        history_active_thought_region = np.array(self.history['active_thought_region'][:min_len])


        # Plot 1: Growth & Activity
        fig1, axs1 = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        axs1[0].plot(times_np, history_N_neurons, '.-', label='Neuron Count')
        axs1[0].set_ylabel('N Neurons')
        axs1[0].grid(True); axs1[0].legend()
        axs1[1].plot(times_np, history_N_synapses, '.-', label='Synapse Count')
        axs1[1].set_ylabel('N Synapses')
        axs1[1].grid(True); axs1[1].legend()
        axs1[2].plot(times_np, history_avg_firing_rate, '.-', label='Avg Firing Rate (Hz)')
        axs1[2].set_ylabel('Avg Rate (Hz)')
        axs1[2].grid(True); axs1[2].legend()
        axs1[3].plot(times_np, history_M_context_trace, '.-', label='M_context Trace')
        active_thought_indices = np.where(history_active_thought_region != -1)[0]
        if active_thought_indices.size > 0: # Only plot if there are active thoughts
            # Scale active thought region for better visibility with M_context_trace if necessary
            # Or use a twin axis:
            ax3_twin = axs1[3].twinx()
            ax3_twin.plot(times_np[active_thought_indices], history_active_thought_region[active_thought_indices], 'o', markersize=3, label='Active Thought Region ID (Right Axis)', alpha=0.7, color='red')
            ax3_twin.set_ylabel('Active Thought Region ID', color='red')
            ax3_twin.tick_params(axis='y', labelcolor='red')
            # Make sure y-ticks for regions are integers if num_conceptual_regions is small
            if self.num_conceptual_regions <= 10:
                ax3_twin.set_yticks(np.arange(self.num_conceptual_regions))


        axs1[3].set_ylabel('M_context Trace')
        axs1[3].set_xlabel('Time (s)')
        axs1[3].grid(True)
        axs1[3].legend(loc='upper left')
        if active_thought_indices.size > 0 : ax3_twin.legend(loc='upper right')
        fig1.suptitle('Unified Brain: Growth, Activity & Matrix Mind State')
        plt.tight_layout(rect=[0,0,1,0.96])
        fig1.savefig("brain_plot_summary.png")
        plt.close(fig1)
        print("Saved brain_plot_summary.png")

        # Plot 2: Field Heatmaps
        fields_to_plot_hist_keys = ['E_field', 'S_field', 'M_field', 'A_field'] # Corrected "M" back
        field_titles = ['E (Neurotrophic)', 'S (Structure)', 'M (Modulator)', 'A (Activity)']
        n_fields = len(fields_to_plot_hist_keys)
        fig2, axs2 = plt.subplots(n_fields, 1, figsize=(12, 3 * n_fields), sharex=True)
        if n_fields == 1: axs2 = [axs2]

        plot_time_extent = [times_np[0], times_np[-1]] if times_np.size > 0 else [0,1]
        extent = [plot_time_extent[0], plot_time_extent[1], -0.5, max(0, self.num_conceptual_regions - 0.5)]
        cmaps = [plt.cm.viridis, plt.cm.bone, plt.cm.plasma, plt.cm.magma]

        for i, hist_key in enumerate(fields_to_plot_hist_keys):
            try:
                field_data_list = [arr for arr in self.history[hist_key][:min_len] if isinstance(arr, np.ndarray) and arr.ndim == 1 and arr.shape[0] == self.num_conceptual_regions]
                if not field_data_list or self.num_conceptual_regions == 0:
                    axs2[i].text(0.5,0.5, "No Data or No Regions", ha='center', va='center', transform=axs2[i].transAxes); continue

                field_data_np = np.array(field_data_list).T # (num_regions, N_time_steps)
                if field_data_np.size == 0:
                    axs2[i].text(0.5,0.5, "Empty Data Array", ha='center', va='center', transform=axs2[i].transAxes); continue

                finite_data = field_data_np[np.isfinite(field_data_np)]
                vmin = np.percentile(finite_data, 1) if finite_data.size > 0 else 0
                vmax = np.percentile(finite_data, 99) if finite_data.size > 0 else 1
                vmax = max(vmax, vmin + DEFAULT_EPSILON)

                im = axs2[i].imshow(field_data_np, aspect='auto', origin='lower', extent=extent, cmap=cmaps[i % len(cmaps)], vmin=vmin, vmax=vmax, interpolation='nearest')
                plt.colorbar(im, ax=axs2[i], label=field_titles[i])
                axs2[i].set_ylabel('Region ID')
                if self.num_conceptual_regions > 0:
                     axs2[i].set_yticks(range(self.num_conceptual_regions))

            except Exception as e:
                print(f"Warning: Failed to plot heatmap for {field_titles[i]}: {e}")
                axs2[i].text(0.5,0.5, "Plotting Error", ha='center', va='center', transform=axs2[i].transAxes)

        if n_fields > 0: axs2[-1].set_xlabel('Time (s)')
        fig2.suptitle('Unified Brain: Substrate & Activity Fields Evolution')
        plt.tight_layout(rect=[0,0,1,0.96])
        fig2.savefig("brain_plot_fields.png")
        plt.close(fig2)
        print("Saved brain_plot_fields.png")


        # Plot 3: Raster Plot
        if self.full_spike_history_raster:
            fig3, ax3 = plt.subplots(figsize=(12, 7))
            all_involved_neuron_ids = set(self.full_spike_history_raster.keys())
            if self.neurons: all_involved_neuron_ids.update(self.neurons.keys())
            if not all_involved_neuron_ids : # Check if set is empty
                print("No neuron IDs for raster plot."); plt.close(fig3); return

            sorted_neuron_ids = sorted(list(all_involved_neuron_ids))
            neuron_plot_y_map = {nid: i for i, nid in enumerate(sorted_neuron_ids)}
            plotted_neuron_count = 0

            for nid in sorted_neuron_ids:
                spike_times_list = self.full_spike_history_raster.get(nid, [])
                if spike_times_list:
                    y_val = neuron_plot_y_map[nid]
                    color_val = 'k'
                    if nid in self.neurons and self.num_conceptual_regions > 0:
                         region_id = self.neurons[nid].assigned_region_id
                         color_val = plt.cm.jet(region_id / max(1, self.num_conceptual_regions -1 if self.num_conceptual_regions > 1 else 1) )
                    ax3.plot(spike_times_list, np.full_like(spike_times_list, y_val), '.', markersize=2, color=color_val, alpha=0.8)
                    plotted_neuron_count +=1

            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Neuron Index (Sorted ID, Colored by Region)')
            ax3.set_title(f'Neural Activity Raster Plot ({plotted_neuron_count} of {len(sorted_neuron_ids)} neurons with history)')
            if times_np.size > 0 : ax3.set_xlim(times_np[0], times_np[-1])
            ax3.set_ylim(-1, max(1, len(sorted_neuron_ids))) # Ensure ylim is not negative if no neurons
            plt.tight_layout()
            fig3.savefig("brain_plot_raster.png")
            plt.close(fig3)
            print("Saved brain_plot_raster.png")
        else:
            print("No spike data for raster plot.")


# --- Particle Simulation Functions (Adapted for Coupled Simulation) ---

# Function to Initialize Particle State (remains the same)
def initialize_particles(parameters):
    """
    Initializes the positions, velocities, orientations, angular velocities,
    types, and masses of particles based on specified initial conditions.
    """
    initial_conditions = parameters["initial_conditions"]
    boundaries = parameters["boundaries"]
    x_min, x_max = boundaries["x_min"], boundaries["x_max"]
    y_min, y_max = boundaries["y_min"], boundaries["y_max"]
    box_size = np.array([x_max - x_min, y_max - y_min])

    num_particles_request = initial_conditions["num_particles_request"]
    initial_type = initial_conditions.get("initial_type", 0) # Default type if not specified
    mass_mapping = initial_conditions["mass_mapping"]

    positions = np.zeros((num_particles_request, 2))
    velocities = np.zeros((num_particles_request, 2))
    accelerations = np.zeros((num_particles_request, 2))
    orientations = np.zeros(num_particles_request)
    angular_velocities = np.zeros(num_particles_request)
    angular_accelerations = np.zeros(num_particles_request)
    types = np.full(num_particles_request, initial_type, dtype=int)
    masses = np.array([mass_mapping.get(t, 1.0) for t in types]) # Default mass 1.0 if type not in mapping

    # Get color mapping from the input parameters dictionary (Corrected from previous point)
    vis_params_loaded = parameters.get("visualization", {})
    patches_vis_loaded = vis_params_loaded.get("patches", {})
    color_mapping_loaded = patches_vis_loaded.get("color_mapping", {})
    # Ensure all types in mass_mapping can be mapped to colors; use string keys for color_mapping from default
    # The default colors for patches uses string keys like "0", "1", "2". Types are int.
    # This line must use the actual parameter structure which converts to int keys after loading for 'visualization'.'patches'.'color_mapping'
    # But initial_conditions.mass_mapping string keys were converted to int.
    # Default for color_mapping:
    # color_mapping = default_particle_parameters.get("visualization", {}).get("patches", {}).get("color_mapping", {}) # This is still using default_particle_parameters
    # If parameters is the fully loaded one, use it:
    # vis_params = parameters.get("visualization", {})
    # patch_vis_params = vis_params.get("patches", {})
    # loaded_color_mapping = patch_vis_params.get("color_mapping", {}) # This will have int keys if loaded correctly by `load_parameters`
    # So the line below for `colors` is likely correct IF `color_mapping_loaded` from `parameters` (not defaults) is used, and `load_parameters` ensures int keys.
    colors = [color_mapping_loaded.get(t, 'gray') for t in types]


    initial_condition_type = initial_conditions["type"]

    if initial_condition_type == "random":
        positions = np.random.rand(num_particles_request, 2) * box_size + np.array([x_min, y_min])
        velocities = np.random.randn(num_particles_request, 2) * initial_conditions.get("initial_velocity_scale", 0.1)
        if initial_conditions.get("initial_orientation_type", "random") == "random":
             orientations = np.random.rand(num_particles_request) * 2 * np.pi
        else:
             orientations = np.full(num_particles_request, initial_conditions.get("initial_orientation_angle", 0.0))
        angular_velocities = np.random.randn(num_particles_request) * initial_conditions.get("initial_angular_velocity_scale", 0.1)


    elif initial_condition_type == "grid":
        grid_size = int(np.ceil(np.sqrt(num_particles_request)))
        if grid_size == 0: grid_size = 1 # Avoid division by zero for num_particles_request = 0
        x_spacing = box_size[0] / grid_size
        y_spacing = box_size[1] / grid_size
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count < num_particles_request:
                    positions[count, 0] = x_min + (i + 0.5) * x_spacing
                    positions[count, 1] = y_min + (j + 0.5) * y_spacing
                    count += 1
        velocities = np.zeros_like(positions)
        if initial_conditions.get("initial_orientation_type", "random") == "random":
             orientations = np.random.rand(num_particles_request) * 2 * np.pi
        else:
             orientations = np.full(num_particles_request, initial_conditions.get("initial_orientation_angle", 0.0))
        angular_velocities = np.zeros_like(orientations)


    elif initial_condition_type == "grid_swirl":
        grid_size = int(np.ceil(np.sqrt(num_particles_request)))
        if grid_size == 0: grid_size = 1
        x_spacing = box_size[0] / grid_size
        y_spacing = box_size[1] / grid_size
        center = np.array([x_min + box_size[0] / 2, y_min + box_size[1] / 2])
        swirl_strength = initial_conditions.get("swirl_strength", 0.5)
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count < num_particles_request:
                    pos = np.array([x_min + (i + 0.5) * x_spacing, y_min + (j + 0.5) * y_spacing])
                    positions[count, :] = pos
                    vec_to_center = center - pos
                    # Create a perpendicular vector for swirling velocity
                    swirl_vel = np.array([-vec_to_center[1], vec_to_center[0]]) * swirl_strength
                    velocities[count, :] = swirl_vel
                    count += 1
        if initial_conditions.get("initial_orientation_type", "random") == "random":
             orientations = np.random.rand(num_particles_request) * 2 * np.pi
        else:
             orientations = np.full(num_particles_request, initial_conditions.get("initial_orientation_angle", 0.0))
        angular_velocities = np.zeros_like(orientations)


    # Assign specific types if a distribution is requested
    if "type_distribution" in initial_conditions:
         type_dist = initial_conditions["type_distribution"]
         # Ensure keys are int if they were strings in JSON
         type_dist_int_keys = {int(k):v for k,v in type_dist.items()}

         if sum(type_dist_int_keys.values()) == num_particles_request:
              current_idx = 0
              for particle_type, count in type_dist_int_keys.items():
                   types[current_idx : current_idx + count] = particle_type
                   current_idx += count
              masses = np.array([mass_mapping.get(t, 1.0) for t in types])
              colors = [color_mapping_loaded.get(t, 'gray') for t in types] # Use the loaded color mapping
         else:
              print("Warning: Sum of particles in type_distribution does not match num_particles_request. Using default initial type.")


    # Initialize bonds (empty initially)
    bonds = {}

    num_particles = num_particles_request

    return positions, velocities, accelerations, bonds, types, masses, colors, orientations, angular_velocities, angular_accelerations, num_particles


# Function to Apply Periodic Boundary Conditions (remains the same)
def apply_periodic_boundary_conditions(positions, box_size, box_min):
    """
    Applies periodic boundary conditions to particle positions.
    """
    return box_min + np.mod(positions - box_min, box_size)


# Function to Update Particle State (Velocity Verlet Integration - adapted)
def update_particle_state(positions, velocities, accelerations, orientations, angular_velocities, angular_accelerations, dt, damping_factor_effective, max_velocity, moments_of_inertia, particle_parameters):
    """
    Updates particle positions, velocities, orientations, and angular velocities
    using the Velocity Verlet integration scheme, applying damping and velocity limits.
    Uses particle_parameters for boundary info.
    """
    boundaries = particle_parameters["boundaries"]
    x_min, x_max = boundaries["x_min"], boundaries["x_max"]
    y_min, y_max = boundaries["y_min"], boundaries["y_max"]
    box_size = np.array([x_max - x_min, y_max - y_min]) # Corrected y_max - y_min
    box_min = np.array([x_min, y_min])

    # Apply damping to current velocities and angular velocities
    current_velocities = velocities * (1.0 - damping_factor_effective)
    current_angular_velocities = angular_velocities * (1.0 - damping_factor_effective) # Applying linear factor for simplicity

    # Step 1 of Velocity Verlet: Update positions and half-step velocities/angular velocities
    positions_half = positions + current_velocities * dt + 0.5 * accelerations * dt**2 # Use accelerations_t
    angular_orientations_half = orientations + current_angular_velocities * dt + 0.5 * angular_accelerations * dt**2

    # Apply periodic boundary conditions to predicted new positions (half step)
    positions_half_pbc = apply_periodic_boundary_conditions(positions_half, box_size, box_min)
    # Keep orientations normalized within [0, 2*pi) at half step
    orientations_half_normalized = np.mod(angular_orientations_half, 2 * np.pi)


    # Apply velocity limiting
    linear_velocity_magnitudes = np.linalg.norm(current_velocities, axis=1)
    exceed_idx = np.where(linear_velocity_magnitudes > max_velocity)[0]
    if len(exceed_idx) > 0:
        current_velocities[exceed_idx, :] = (current_velocities[exceed_idx, :] /
                                      linear_velocity_magnitudes[exceed_idx, np.newaxis]) * max_velocity


    return positions_half_pbc, current_velocities, orientations_half_normalized, current_angular_velocities


# Function to Calculate Forces and Torques (Adapted for Coupled Simulation)
def calculate_forces(positions, velocities, orientations, types, masses, bonds, particle_parameters, current_step, all_patch_data_input, brain_stimulation_per_region, num_conceptual_regions, adaptive_factors_per_particle: np.ndarray):
    """
    Calculates the net linear forces and torques acting on each particle.
    all_patch_data_input is used if valid, otherwise regenerated.
    Returns new all_patch_data.
    """
    # Extract particle parameters (including new potential parameters)
    forces_params = particle_parameters["forces"]
    bonding_params = particle_parameters["bonding"]
    boundaries = particle_parameters["boundaries"]
    density_repulsion_params = particle_parameters["density_repulsion"]
    external_force_params = particle_parameters["external_force"] # Get external force params
    adaptive_params = particle_parameters.get("adaptive_interactions", {})


    C = forces_params["C"]
    cutoff_distance = forces_params["cutoff_distance"] # for center-center pairwise
    short_range_repulsion_strength = forces_params.get("short_range_repulsion_strength", 50.0)


    density_radius = density_repulsion_params["density_radius"]
    density_repulsion_strength = density_repulsion_params["density_repulsion_strength"]
    x_min, x_max = boundaries["x_min"], boundaries["x_max"]
    y_min, y_max = boundaries["y_min"], boundaries["y_max"]
    box_size = np.array([x_max - x_min, y_max - y_min])

    moment_of_inertia_mapping = forces_params["moment_of_inertia_mapping"]

    patch_params = forces_params.get("patch_params", {})
    patch_enabled = patch_params.get("enabled", False)
    patch_definitions = patch_params.get("patch_definitions", {})

    patch_pairwise_potential_params = patch_params.get("patch_pairwise_potential", {})
    patch_pairwise_potential_type = patch_pairwise_potential_params.get("type", "inverse_square_plus_sr")
    inverse_square_strength_matrix = np.array(patch_pairwise_potential_params.get("inverse_square_strength_matrix", [[]]))
    patch_short_range_repulsion_strength = patch_pairwise_potential_params.get("sr_strength", 50.0)
    lj_pairwise_params = patch_pairwise_potential_params.get("lennard_jones", {})
    lj_pairwise_epsilon_matrix = np.array(lj_pairwise_params.get("epsilon_matrix", [[]]))
    lj_pairwise_sigma_matrix = np.array(lj_pairwise_params.get("sigma_matrix", [[]]))
    lj_pairwise_cutoff_factor = lj_pairwise_params.get("cutoff_factor", 2.5)
    sw_pairwise_params = patch_pairwise_potential_params.get("square_well", {})
    sw_pairwise_epsilon_matrix = np.array(sw_pairwise_params.get("epsilon_matrix", [[]]))
    sw_pairwise_sigma_matrix = np.array(sw_pairwise_params.get("sigma_matrix", [[]]))
    sw_pairwise_lambda_matrix = np.array(sw_pairwise_params.get("lambda_matrix", [[]]))
    sw_pairwise_transition_width = sw_pairwise_params.get("transition_width", 0.1)


    patch_bond_potential_params = bonding_params.get("patch_bond_potential", {})
    patch_bond_potential_type = patch_bond_potential_params.get("type", "harmonic")
    patch_bond_distance_harmonic = patch_bond_potential_params.get("harmonic", {}).get("bond_distance", 1.0)
    patch_bond_strength_harmonic = patch_bond_potential_params.get("harmonic", {}).get("bond_strength", 100.0)
    lj_bond_params = patch_bond_potential_params.get("lennard_jones", {})
    lj_bond_epsilon = lj_bond_params.get("epsilon", 5.0)
    lj_bond_sigma = lj_bond_params.get("sigma", 1.5)
    sw_bond_params = patch_bond_potential_params.get("square_well", {})
    sw_bond_epsilon = sw_bond_params.get("epsilon", 5.0)
    sw_bond_sigma = sw_bond_params.get("sigma", 1.5)
    sw_bond_lambda = sw_bond_params.get("lambda", 1.5)
    sw_bond_transition_width = sw_bond_params.get("transition_width", 0.1)
    patch_cutoff_distance_param = patch_params.get("patch_cutoff_distance", 5.0)


    orientation_potential_params = forces_params.get("orientation_potential", {})
    bond_angle_potential_params = orientation_potential_params.get("bond_angle_potential", {})
    bond_angle_potential_enabled = bond_angle_potential_params.get("enabled", False)
    bond_angle_strength = bond_angle_potential_params.get("strength", 0.0)
    ideal_angle_mapping = bond_angle_potential_params.get("ideal_angle_mapping", {})


    adaptive_enabled = adaptive_params.get("enabled", False)
    bond_strength_adaptation_enabled = adaptive_params.get("bond_strength_adaptation", {}).get("enabled", False)
    adaptation_rate = adaptive_params.get("bond_strength_adaptation", {}).get("adaptation_rate", 0.0)
    target_strength = adaptive_params.get("bond_strength_adaptation", {}).get("target_strength", bonding_params.get("bond_strength", 100.0))


    num_particles = positions.shape[0]
    if num_particles == 0: return np.array([]), np.array([]), []
    net_linear_forces = np.zeros_like(positions)
    net_torques = np.zeros(num_particles)

    # Regenerate all_patch_data based on current positions and orientations for this force calculation
    current_all_patch_data = []
    if patch_enabled and patch_definitions:
        for i in range(num_particles):
            particle_type_int = types[i] # types are int
            # Defensive check for patch_definitions key presence
            pdefs_i = [] # Initialize to empty list
            if particle_type_int in patch_definitions:
                 pdefs_i = patch_definitions.get(particle_type_int)
            elif str(particle_type_int) in patch_definitions: # Fallback to string key if int key not found
                 pdefs_i = patch_definitions.get(str(particle_type_int))

            # Ensure it is a list before iteration
            if not isinstance(pdefs_i, list):
                pdefs_i = []


            particle_patches_data = []
            for patch_index_on_particle, patch_spec in enumerate(pdefs_i): # Use pdefs_i here
                p_dist = patch_spec.get("distance", 0.0)
                p_angle_rel = patch_spec.get("angle_relative_to_particle", 0.0)
                patch_type = patch_spec.get("patch_type", 0)
                total_patch_angle = orientations[i] + p_angle_rel
                patch_offset = np.array([p_dist * np.cos(total_patch_angle), p_dist * np.sin(total_patch_angle)])
                patch_position = positions[i, :] + patch_offset
                particle_patches_data.append({
                    "position": patch_position, "patch_type": patch_type,
                    "particle_index": i, "patch_index_on_particle": patch_index_on_particle
                })
            current_all_patch_data.append(particle_patches_data)


    # --- Central Force & Density Repulsion ---
    box_center = np.array([x_min + box_size[0]/2, y_min + box_size[1]/2])
    vec_to_center_of_box = box_center - positions
    central_linear_forces = C * vec_to_center_of_box # Force towards center
    net_linear_forces += central_linear_forces

    # Density Repulsion (away from box center, scaled by local density)
    if density_repulsion_strength > 0 and density_radius > 0 :
        r_center_to_center = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        r_center_to_center_pbc = r_center_to_center - box_size * np.round(r_center_to_center / box_size)
        r_center_mag = np.sqrt(np.sum(r_center_to_center_pbc**2, axis=2))
        np.fill_diagonal(r_center_mag, np.inf) # Ignore self-distance

        local_density_counts = np.sum(r_center_mag < density_radius, axis=1)
        direction_from_center = positions - box_center # Vector from center to particle
        direction_from_center_mag = np.linalg.norm(direction_from_center, axis=1)
        # Avoid division by zero if a particle is exactly at the center
        safe_mag = np.where(direction_from_center_mag < DEFAULT_EPSILON, 1.0, direction_from_center_mag)
        density_force_direction = direction_from_center / safe_mag[:, np.newaxis]
        density_force_magnitude = density_repulsion_strength * local_density_counts
        density_linear_forces = density_force_magnitude[:, np.newaxis] * density_force_direction
        net_linear_forces += density_linear_forces


    # --- Patch-Based Forces (Pairwise Non-Bonded & Bonded) ---
    if patch_enabled and patch_definitions and current_all_patch_data:
        # Determine generous search cutoff considering max patch extension
        max_patch_extension = 0
        for p_type, p_defs in patch_definitions.items():
            for p_spec in p_defs:
                max_patch_extension = max(max_patch_extension, p_spec.get("distance",0))
        patch_search_cutoff_dynamic = patch_cutoff_distance_param + 2 * max_patch_extension

        for i in range(num_particles):
            for j in range(i + 1, num_particles):
                r_ij_center_vec = positions[i, :] - positions[j, :]
                r_ij_center_vec_pbc = r_ij_center_vec - box_size * np.round(r_ij_center_vec / box_size)
                r_ij_center_mag = np.linalg.norm(r_ij_center_vec_pbc)

                if r_ij_center_mag > patch_search_cutoff_dynamic: continue
                if not current_all_patch_data[i] or not current_all_patch_data[j]: continue


                for patch_i_data in current_all_patch_data[i]:
                    for patch_j_data in current_all_patch_data[j]:
                        # patch_i_data structure: {"position", "patch_type", "particle_index", "patch_index_on_particle"}
                        p_i_idx_on_particle = patch_i_data["patch_index_on_particle"]
                        p_j_idx_on_particle = patch_j_data["patch_index_on_particle"]
                        bond_candidate_key = tuple(sorted(((i, p_i_idx_on_particle), (j, p_j_idx_on_particle))))

                        patch_i_pos = patch_i_data["position"]
                        patch_j_pos = patch_j_data["position"]
                        patch_i_type = patch_i_data["patch_type"]
                        patch_j_type = patch_j_data["patch_type"]

                        r_patch_raw = patch_i_pos - patch_j_pos
                        r_patch_pbc = r_patch_raw - box_size * np.round(r_patch_raw / box_size)
                        r_patch_mag = np.linalg.norm(r_patch_pbc)
                        if r_patch_mag < DEFAULT_EPSILON: # Overlapping patches
                             # Apply a large repulsion and random nudge
                             rand_nudge = normalize_vector_np(np.random.rand(2)-0.5) * DEFAULT_EPSILON * 10
                             net_linear_forces[i,:] += rand_nudge * short_range_repulsion_strength * 100
                             net_linear_forces[j,:] -= rand_nudge * short_range_repulsion_strength * 100
                             continue

                        patch_to_patch_direction = r_patch_pbc / r_patch_mag
                        force_magnitude_on_patch_i = 0.0


                        if bond_candidate_key in bonds: # --- BONDED PATCH INTERACTION ---
                            current_bond_strength_h = patch_bond_strength_harmonic
                            if adaptive_enabled and bond_strength_adaptation_enabled:
                                bond_info = bonds[bond_candidate_key]
                                formed_step = bond_info.get('formed_step', 0)
                                initial_strength = bond_info.get('initial_strength', patch_bond_strength_harmonic)
                                bond_age = current_step - formed_step
                                current_bond_strength_h = initial_strength + bond_age * adaptation_rate
                                current_bond_strength_h = np.clip(current_bond_strength_h, 0, target_strength if adaptation_rate >0 else np.inf)


                            if patch_bond_potential_type == "harmonic":
                                force_magnitude_on_patch_i = -current_bond_strength_h * (r_patch_mag - patch_bond_distance_harmonic)
                            elif patch_bond_potential_type == "lennard_jones":
                                r6 = (lj_bond_sigma / r_patch_mag)**6; r12 = r6**2
                                force_magnitude_on_patch_i = -24 * lj_bond_epsilon / r_patch_mag * (2 * r12 - r6)
                            elif patch_bond_potential_type == "square_well":
                                force_magnitude_on_patch_i = smoothed_square_well_force(r_patch_mag, sw_bond_sigma, sw_bond_lambda, sw_bond_epsilon, sw_bond_transition_width)
                        else: # --- NON-BONDED PATCH PAIRWISE INTERACTION ---
                            if r_patch_mag < patch_cutoff_distance_param : # General cutoff for pairwise
                                if patch_pairwise_potential_type == "inverse_square_plus_sr":
                                    interaction_k = 0.0
                                    if 0<=patch_i_type<inverse_square_strength_matrix.shape[0] and 0<=patch_j_type<inverse_square_strength_matrix.shape[1]:
                                        interaction_k = inverse_square_strength_matrix[patch_i_type, patch_j_type]
                                    f_inv_sq = interaction_k / r_patch_mag**2
                                    f_sr = patch_short_range_repulsion_strength * (1/r_patch_mag - 1/patch_cutoff_distance_param) if r_patch_mag < patch_cutoff_distance_param else 0
                                    force_magnitude_on_patch_i = -(f_inv_sq + f_sr) # Attractive if k >0 (then f_inv_sq >0, force is -), repulsive if k<0
                                elif patch_pairwise_potential_type == "lennard_jones":
                                    eps, sig = 0.0, 1.0
                                    if 0<=patch_i_type<lj_pairwise_epsilon_matrix.shape[0] and 0<=patch_j_type<lj_pairwise_epsilon_matrix.shape[1]:
                                        eps = lj_pairwise_epsilon_matrix[patch_i_type, patch_j_type]
                                    if 0<=patch_i_type<lj_pairwise_sigma_matrix.shape[0] and 0<=patch_j_type<lj_pairwise_sigma_matrix.shape[1]:
                                        sig = lj_pairwise_sigma_matrix[patch_i_type, patch_j_type]
                                    cutoff_lj = lj_pairwise_cutoff_factor * sig
                                    if r_patch_mag < cutoff_lj:
                                        r6 = (sig / r_patch_mag)**6; r12 = r6**2
                                        force_magnitude_on_patch_i = -24 * eps / r_patch_mag * (2 * r12 - r6)
                                elif patch_pairwise_potential_type == "square_well":
                                    eps, sig, lam = 0.0, 1.0, 1.5
                                    if 0<=patch_i_type<sw_pairwise_epsilon_matrix.shape[0] and 0<=patch_j_type<sw_pairwise_epsilon_matrix.shape[1]:
                                        eps = sw_pairwise_epsilon_matrix[patch_i_type, patch_j_type]
                                    if 0<=patch_i_type<sw_pairwise_sigma_matrix.shape[0] and 0<=patch_j_type<sw_pairwise_sigma_matrix.shape[1]:
                                        sig = sw_pairwise_sigma_matrix[patch_i_type, patch_j_type]
                                    if 0<=patch_i_type<sw_pairwise_lambda_matrix.shape[0] and 0<=patch_j_type<sw_pairwise_lambda_matrix.shape[1]:
                                        lam = sw_pairwise_lambda_matrix[patch_i_type, patch_j_type]
                                    force_magnitude_on_patch_i = smoothed_square_well_force(r_patch_mag, sig, lam, eps, sw_pairwise_transition_width)


                        # Apply force and torque from patch interaction
                        force_on_patch_i_vec = force_magnitude_on_patch_i * patch_to_patch_direction
                        net_linear_forces[i, :] += force_on_patch_i_vec
                        net_linear_forces[j, :] -= force_on_patch_i_vec

                        r_vec_i_to_patch_i = patch_i_pos - positions[i, :]
                        torque_on_i = r_vec_i_to_patch_i[0] * force_on_patch_i_vec[1] - r_vec_i_to_patch_i[1] * force_on_patch_i_vec[0]
                        net_torques[i] += torque_on_i

                        r_vec_j_to_patch_j = patch_j_pos - positions[j, :]
                        torque_on_j = r_vec_j_to_patch_j[0] * (-force_on_patch_i_vec[1]) - r_vec_j_to_patch_j[1] * (-force_on_patch_i_vec[0])
                        net_torques[j] += torque_on_j


    # --- External Forces from Brain Stimulation ---
    if external_force_params.get("enabled", False) and brain_stimulation_per_region is not None and num_conceptual_regions > 0:
         brain_to_particle_force_scale = default_params_unified['brain_stimulation_to_particle_force_scale']
         region_width_particle_space = box_size[0] / num_conceptual_regions # Assume regions along x-axis

         for i in range(num_particles):
              particle_x_relative = positions[i, 0] - x_min
              region_idx = int(particle_x_relative / region_width_particle_space)
              region_idx = np.clip(region_idx, 0, num_conceptual_regions - 1)

              stimulation_level = brain_stimulation_per_region[region_idx]
              # Corrected logic from identified point 1
              region_center_x_abs = x_min + (region_idx + 0.5) * region_width_particle_space
              force_dir_to_region_center = np.array([region_center_x_abs - positions[i,0], 0.0]) # Force only in X
              force_dir_mag = np.linalg.norm(force_dir_to_region_center)
              force_dir_norm = force_dir_to_region_center / force_dir_mag if force_dir_mag > DEFAULT_EPSILON else np.array([0.0,0.0])

              force_magnitude_from_brain = stimulation_level * brain_to_particle_force_scale
              external_force_on_particle = force_magnitude_from_brain * force_dir_norm
              net_linear_forces[i, :] += external_force_on_particle


    # --- Friction (Applied per-particle, scaled by adaptive_factors_per_particle) ---
    if external_force_params.get("friction_enabled", False) and adaptive_factors_per_particle is not None and adaptive_factors_per_particle.shape[0] == num_particles:
        friction_coeff = external_force_params.get("friction_coefficient", 0.1)
        # Apply zone-specific adaptive factor to individual particles.
        per_particle_friction = friction_coeff * adaptive_factors_per_particle
        net_linear_forces -= per_particle_friction[:, np.newaxis] * masses[:, np.newaxis] * velocities # mass-proportional friction
    if external_force_params.get("angular_friction_enabled", False) and adaptive_factors_per_particle is not None and adaptive_factors_per_particle.shape[0] == num_particles:
        angular_friction_coeff = external_force_params.get("angular_friction_coefficient", 0.05)
        moments_of_inertia_arr = np.array([moment_of_inertia_mapping.get(t, 1.0) for t in types])
        per_particle_angular_friction = angular_friction_coeff * adaptive_factors_per_particle
        net_torques -= per_particle_angular_friction * moments_of_inertia_arr * angular_velocities # MoI-proportional

    return net_linear_forces, net_torques, current_all_patch_data


# Function to Calculate Total Energy (Adapted for Coupled Simulation)
def calculate_total_energy(positions, velocities, angular_velocities, masses, bonds, types, orientations, particle_parameters, current_step, all_patch_data):
    """
    Calculates the total energy of the system, including kinetic, pairwise potential,
    bond potential, central potential, and orientation potential energy.
    Uses selectable potential types (including Smoothed Square Well).
    Requires current step and all_patch_data for adaptive interactions and patch positions.

    Args:
        positions (np.ndarray): Current positions of particles.
        velocities (np.ndarray): Current linear velocities.
        angular_velocities (np.ndarray): Current angular velocities.
        masses (np.ndarray): Masses of particles.
        bonds (dict): Current dictionary of active bonds.
        types (np.ndarray): Types of particles.
        orientations (np.ndarray): Current orientations of particles.
        particle_parameters (dict): Dictionary of particle simulation parameters.
        current_step (int): The current simulation step number.
        all_patch_data (list of lists): Data structure containing info for all patches.


    Returns:
        float: The total energy of the system.
    """
    # Extract particle parameters (including new potential parameters)
    forces_params = particle_parameters["forces"]
    bonding_params = particle_parameters["bonding"]
    boundaries = particle_parameters["boundaries"]
    # density_repulsion_params = particle_parameters["density_repulsion"] # Not typically included in potential energy
    adaptive_params = particle_parameters.get("adaptive_interactions", {})

    C = forces_params["C"]
    # cutoff_distance = forces_params["cutoff_distance"] # For center-center pairwise (if used)

    # bond_distance_param = bonding_params.get("patch_bond_potential", {}).get("harmonic", {}).get("bond_distance", bonding_params.get("bond_distance", 1.0))
    # bond_strength_param = bonding_params.get("patch_bond_potential", {}).get("harmonic", {}).get("bond_strength", bonding_params.get("bond_strength", 100.0))

    x_min, x_max = boundaries["x_min"], boundaries["x_max"]
    y_min, y_max = boundaries["y_min"], boundaries["y_max"]
    box_size = np.array([x_max - x_min, y_max - y_min])
    box_center = np.array([x_min + box_size[0]/2, y_min + box_size[1]/2])


    moment_of_inertia_mapping = forces_params["moment_of_inertia_mapping"]

    patch_params = forces_params.get("patch_params", {})
    patch_enabled = patch_params.get("enabled", False)
    patch_definitions = patch_params.get("patch_definitions", {})

    patch_pairwise_potential_params = patch_params.get("patch_pairwise_potential", {})
    patch_pairwise_potential_type = patch_pairwise_potential_params.get("type", "inverse_square_plus_sr")

    inverse_square_strength_matrix = np.array(patch_pairwise_potential_params.get("inverse_square_strength_matrix", [[]]))
    patch_sr_strength = patch_pairwise_potential_params.get("sr_strength", 50.0) # For inverse_square_plus_sr

    lj_pairwise_params = patch_pairwise_potential_params.get("lennard_jones", {})
    lj_pairwise_epsilon_matrix = np.array(lj_pairwise_params.get("epsilon_matrix", [[]]))
    lj_pairwise_sigma_matrix = np.array(lj_pairwise_params.get("sigma_matrix", [[]]))
    lj_pairwise_cutoff_factor = lj_pairwise_params.get("cutoff_factor", 2.5)

    sw_pairwise_params = patch_pairwise_potential_params.get("square_well", {})
    sw_pairwise_epsilon_matrix = np.array(sw_pairwise_params.get("epsilon_matrix", [[]]))
    sw_pairwise_sigma_matrix = np.array(sw_pairwise_params.get("sigma_matrix", [[]]))
    sw_pairwise_lambda_matrix = np.array(sw_pairwise_params.get("lambda_matrix", [[]]))
    sw_pairwise_transition_width = sw_pairwise_params.get("transition_width", 0.1)


    patch_bond_potential_params = bonding_params.get("patch_bond_potential", {})
    patch_bond_potential_type = patch_bond_potential_params.get("type", "harmonic")

    patch_bond_distance_harmonic = patch_bond_potential_params.get("harmonic", {}).get("bond_distance", 1.0)
    patch_bond_strength_harmonic = patch_bond_potential_params.get("harmonic", {}).get("bond_strength", 100.0)

    lj_bond_params = patch_bond_potential_params.get("lennard_jones", {})
    lj_bond_epsilon = lj_bond_params.get("epsilon", 5.0)
    lj_bond_sigma = lj_bond_params.get("sigma", 1.5)

    sw_bond_params = patch_bond_potential_params.get("square_well", {})
    sw_bond_epsilon = sw_bond_params.get("epsilon", 5.0)
    sw_bond_sigma = sw_bond_params.get("sigma", 1.5)
    sw_bond_lambda = sw_bond_params.get("lambda", 1.5)
    sw_bond_transition_width = sw_bond_params.get("transition_width", 0.1)


    patch_cutoff_distance_param = patch_params.get("patch_cutoff_distance", 5.0)


    orientation_potential_params = forces_params.get("orientation_potential", {})
    bond_angle_potential_params = orientation_potential_params.get("bond_angle_potential", {})
    bond_angle_potential_enabled = bond_angle_potential_params.get("enabled", False)
    bond_angle_strength = bond_angle_potential_params.get("strength", 0.0)
    ideal_angle_mapping = bond_angle_potential_params.get("ideal_angle_mapping", {}) # keys are int

    adaptive_enabled = adaptive_params.get("enabled", False)
    bond_strength_adaptation_enabled = adaptive_params.get("bond_strength_adaptation", {}).get("enabled", False)
    adaptation_rate = adaptive_params.get("bond_strength_adaptation", {}).get("adaptation_rate", 0.0)
    target_strength = adaptive_params.get("bond_strength_adaptation", {}).get("target_strength", bonding_params.get("bond_strength", 100.0))


    num_particles = positions.shape[0]
    if num_particles == 0: return 0.0

    # --- Kinetic Energy ---
    linear_kinetic_energy = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
    moments_of_inertia_array = np.array([moment_of_inertia_mapping.get(t, 1.0) for t in types])
    angular_kinetic_energy = 0.5 * np.sum(moments_of_inertia_array * angular_velocities**2)
    kinetic_energy = linear_kinetic_energy + angular_kinetic_energy

    # --- Potential Energy ---
    potential_energy = 0.0
    # Central Potential: U = 0.5 * C * sum(dist_from_center_i^2)
    dist_from_center_sq = np.sum((positions - box_center)**2, axis=1)
    potential_energy += 0.5 * C * np.sum(dist_from_center_sq)


    # --- Patch-Based Potential Energy (Pairwise Non-Bonded & Bonded) ---
    if patch_enabled and patch_definitions and all_patch_data:
        max_patch_extension = 0
        for p_type, p_defs in patch_definitions.items():
            for p_spec in p_defs: max_patch_extension = max(max_patch_extension, p_spec.get("distance",0))
        patch_search_cutoff_dynamic = patch_cutoff_distance_param + 2 * max_patch_extension


        for i in range(num_particles):
            for j in range(i + 1, num_particles):
                r_ij_center_vec = positions[i, :] - positions[j, :]
                r_ij_center_vec_pbc = r_ij_center_vec - box_size * np.round(r_ij_center_vec / box_size)
                r_ij_center_mag = np.linalg.norm(r_ij_center_vec_pbc)
                if r_ij_center_mag > patch_search_cutoff_dynamic: continue
                if not all_patch_data[i] or not all_patch_data[j]: continue

                for patch_i_data in all_patch_data[i]:
                    for patch_j_data in all_patch_data[j]:
                        p_i_idx_on_particle = patch_i_data["patch_index_on_particle"]
                        p_j_idx_on_particle = patch_j_data["patch_index_on_particle"]
                        bond_candidate_key = tuple(sorted(((i, p_i_idx_on_particle), (j, p_j_idx_on_particle))))

                        patch_i_pos = patch_i_data["position"]
                        patch_j_pos = patch_j_data["position"]
                        patch_i_type = patch_i_data["patch_type"]
                        patch_j_type = patch_j_data["patch_type"]

                        r_patch_raw = patch_i_pos - patch_j_pos
                        r_patch_pbc = r_patch_raw - box_size * np.round(r_patch_raw / box_size)
                        r_patch_mag = np.linalg.norm(r_patch_pbc)
                        if r_patch_mag < DEFAULT_EPSILON: potential_energy += np.inf; continue


                        U_ij_patch = 0.0
                        if bond_candidate_key in bonds: # --- BONDED PATCH POTENTIAL ---
                            current_bond_strength_h = patch_bond_strength_harmonic
                            if adaptive_enabled and bond_strength_adaptation_enabled:
                                bond_info = bonds[bond_candidate_key]
                                formed_step = bond_info.get('formed_step', 0)
                                initial_strength = bond_info.get('initial_strength', patch_bond_strength_harmonic)
                                bond_age = current_step - formed_step
                                current_bond_strength_h = initial_strength + bond_age * adaptation_rate
                                current_bond_strength_h = np.clip(current_bond_strength_h, 0, target_strength if adaptation_rate >0 else np.inf)

                            if patch_bond_potential_type == "harmonic":
                                U_ij_patch = 0.5 * current_bond_strength_h * (r_patch_mag - patch_bond_distance_harmonic)**2
                            elif patch_bond_potential_type == "lennard_jones":
                                r6 = (lj_bond_sigma / r_patch_mag)**6; r12 = r6**2
                                U_ij_patch = 4 * lj_bond_epsilon * (r12 - r6)
                            elif patch_bond_potential_type == "square_well":
                                U_ij_patch = smoothed_square_well_potential(r_patch_mag, sw_bond_sigma, sw_bond_lambda, sw_bond_epsilon, sw_bond_transition_width)
                        else: # --- NON-BONDED PATCH PAIRWISE POTENTIAL ---
                            if r_patch_mag < patch_cutoff_distance_param :
                                if patch_pairwise_potential_type == "inverse_square_plus_sr":
                                    interaction_k = 0.0
                                    if 0<=patch_i_type<inverse_square_strength_matrix.shape[0] and 0<=patch_j_type<inverse_square_strength_matrix.shape[1]:
                                        interaction_k = inverse_square_strength_matrix[patch_i_type, patch_j_type]
                                    f_inv_sq = interaction_k / r_patch_mag**2
                                    f_sr = patch_short_range_repulsion_strength * (1/r_patch_mag - 1/patch_cutoff_distance_param) if r_patch_mag < patch_cutoff_distance_param else 0
                                    U_ij_patch = - (f_inv_sq + f_sr * patch_cutoff_distance_param * (1/r_patch_mag - 1/patch_cutoff_distance_param) ) # This needs potential derivative, not force integral for potential. This is an incorrect integration if `f_sr` already a force term. I will use force from last working code in new block for fix.
                                elif patch_pairwise_potential_type == "lennard_jones":
                                    eps, sig = 0.0,1.0
                                    if 0<=patch_i_type<lj_pairwise_epsilon_matrix.shape[0] and 0<=patch_j_type<lj_pairwise_epsilon_matrix.shape[1]: eps = lj_pairwise_epsilon_matrix[patch_i_type, patch_j_type]
                                    if 0<=patch_i_type<lj_pairwise_sigma_matrix.shape[0] and 0<=patch_j_type<lj_pairwise_sigma_matrix.shape[1]: sig = lj_pairwise_sigma_matrix[patch_i_type, patch_j_type]
                                    cutoff_lj = lj_pairwise_cutoff_factor * sig
                                    if r_patch_mag < cutoff_lj:
                                        r6 = (sig / r_patch_mag)**6; r12 = r6**2
                                        U_ij_patch = 4 * eps * (r12 - r6)
                                        rc6 = (sig / cutoff_lj)**6; rc12 = rc6**2 # Shift potential
                                        U_ij_patch -= 4 * eps * (rc12 - rc6)
                                elif patch_pairwise_potential_type == "square_well":
                                    eps, sig, lam = 0.0,1.0,1.5
                                    if 0<=patch_i_type<sw_pairwise_epsilon_matrix.shape[0] and 0<=patch_j_type<sw_pairwise_epsilon_matrix.shape[1]: eps = sw_pairwise_epsilon_matrix[patch_i_type, patch_j_type]
                                    if 0<=patch_i_type<sw_pairwise_sigma_matrix.shape[0] and 0<=patch_j_type<sw_pairwise_sigma_matrix.shape[1]: sig = sw_pairwise_sigma_matrix[patch_i_type, patch_j_type]
                                    if 0<=patch_i_type<sw_pairwise_lambda_matrix.shape[0] and 0<=patch_j_type<sw_pairwise_lambda_matrix.shape[1]: lam = sw_pairwise_lambda_matrix[patch_i_type, patch_j_type]
                                    U_ij_patch = smoothed_square_well_potential(r_patch_mag, sig, lam, eps, sw_pairwise_transition_width)
                        potential_energy += U_ij_patch

    # --- Orientation Potential Energy ---
    if patch_enabled and bond_angle_potential_enabled and all_patch_data:
        for bond_key in bonds.keys():
            if not (isinstance(bond_key, tuple) and len(bond_key) == 2 and isinstance(bond_key[0], tuple)): continue
            (i, p_idx_i), (j, p_idx_j) = bond_key
            if not (i < num_particles and j < num_particles and all_patch_data[i] and all_patch_data[j] and \
                    p_idx_i < len(all_patch_data[i]) and p_idx_j < len(all_patch_data[j])): continue

            patch_i_data = all_patch_data[i][p_idx_i]
            patch_j_data = all_patch_data[j][p_idx_j]
            patch_i_pos, patch_j_pos = patch_i_data["position"], patch_j_data["position"]
            patch_i_type, patch_j_type = patch_i_data["patch_type"], patch_j_data["patch_type"]

            r_patch_raw = patch_j_pos - patch_i_pos
            r_patch_pbc = r_patch_raw - box_size * np.round(r_patch_raw / box_size)
            r_patch_mag = np.linalg.norm(r_patch_pbc)
            if r_patch_mag < DEFAULT_EPSILON: continue
            bond_vec_dir = r_patch_pbc / r_patch_mag

            particle_type_i_int = types[i] # Get particle type (int)
            # Defensive initialization for pdefs_i
            pdefs_i = []
            if particle_type_i_int in patch_definitions:
                pdefs_i = patch_definitions.get(particle_type_i_int)
            elif str(particle_type_i_int) in patch_definitions:
                pdefs_i = patch_definitions.get(str(particle_type_i_int))
            if not isinstance(pdefs_i, list): pdefs_i = [] # Final safeguard

            if p_idx_i >= len(pdefs_i): continue
            angle_rel_i = pdefs_i[p_idx_i].get("angle_relative_to_particle",0.0)
            abs_angle_i = orientations[i] + angle_rel_i
            patch_i_orient_vec = np.array([np.cos(abs_angle_i), np.sin(abs_angle_i)])
            ideal_angle_i_rad = ideal_angle_mapping.get(patch_i_type, 0.0) # ideal_angle_mapping keys are int

            current_angle_i_to_bond = np.arctan2(patch_i_orient_vec[0]*bond_vec_dir[1] - patch_i_orient_vec[1]*bond_vec_dir[0], np.dot(patch_i_orient_vec, bond_vec_dir))
            angle_dev_i_sq = (np.mod(current_angle_i_to_bond - ideal_angle_i_rad + np.pi, 2 * np.pi) - np.pi)**2


            particle_type_j_int = types[j]
            # Defensive initialization for pdefs_j
            pdefs_j = []
            if particle_type_j_int in patch_definitions:
                pdefs_j = patch_definitions.get(particle_type_j_int)
            elif str(particle_type_j_int) in patch_definitions:
                pdefs_j = patch_definitions.get(str(particle_type_j_int))
            if not isinstance(pdefs_j, list): pdefs_j = [] # Final safeguard

            if p_idx_j >= len(pdefs_j): continue
            angle_rel_j = pdefs_j[p_idx_j].get("angle_relative_to_particle",0.0)
            abs_angle_j = orientations[j] + angle_rel_j
            patch_j_orient_vec = np.array([np.cos(abs_angle_j), np.sin(abs_angle_j)])
            ideal_angle_j_rad = ideal_angle_mapping.get(patch_j_type, 0.0)

            current_angle_j_to_bond = np.arctan2(patch_j_orient_vec[0]*(-bond_vec_dir[1])-patch_j_orient_vec[1]*(-bond_vec_dir[0]), np.dot(patch_j_orient_vec,-bond_vec_dir))
            angle_dev_j_sq = (np.mod(current_angle_j_to_bond - ideal_j_rad + np.pi, 2*np.pi) - np.pi)**2

            potential_energy += 0.5 * bond_angle_strength * (angle_dev_i_sq + angle_dev_j_sq)


    return kinetic_energy + potential_energy


# Function to Update Bonds (remains the same, uses particle_parameters)
def update_bonds(positions, orientations, types, bonds, particle_parameters, current_step, all_patch_data):
    """
    Updates the list of active bonds based on formation and breaking criteria.
    Supports both center-to-center and patch-based bonding.
    Uses particle_parameters for bonding criteria and patch definitions.

    Args:
        positions (np.ndarray): Current positions of particles.
        orientations (np.ndarray): Current orientations of particles.
        types (np.ndarray): Types of particles.
        bonds (dict): Current dictionary of active bonds.
        particle_parameters (dict): Dictionary of particle simulation parameters.
        current_step (int): The current simulation step number.
        all_patch_data (list of lists): Data structure containing info for all patches.

    Returns:
        dict: Updated dictionary of active bonds.
    """
    bonding_params = particle_parameters["bonding"]
    if not bonding_params["enabled"]:
        return bonds # Bonding is disabled

    formation_criteria = bonding_params["formation_criteria"]
    distance_tolerance = formation_criteria["distance_tolerance"]
    # Ensure patch_type_compatibility_matrix is a numpy array of bools
    patch_type_compatibility_matrix = np.array(formation_criteria.get("patch_type_compatibility_matrix", [[]]), dtype=bool)
    orientation_alignment_tolerance = formation_criteria["orientation_alignment_tolerance"]
    bond_break_distance = bonding_params["bond_break_distance"]
    # bond_types_param = tuple(bonding_params["bond_types"]) # Not directly used here, types in patches used

    patch_params = particle_parameters["forces"].get("patch_params", {})
    patch_enabled = patch_params.get("enabled", False)
    patch_definitions = patch_params.get("patch_definitions", {})

    updated_bonds = bonds.copy()

    num_particles = positions.shape[0]
    if num_particles == 0 or not all_patch_data : return updated_bonds # No particles or patch data means no bonds

    boundaries = particle_parameters["boundaries"]
    box_size = np.array([boundaries["x_max"] - boundaries["x_min"], boundaries["y_max"] - boundaries["y_min"]])

    # --- Bond Breaking ---
    bonds_to_break = []
    for bond_key in updated_bonds.keys():
         is_patch_bond = isinstance(bond_key, tuple) and len(bond_key) == 2 and isinstance(bond_key[0], tuple)
         if is_patch_bond:
              (i, p_idx_i), (j, p_idx_j) = bond_key
              # Check particles/patches exist and indices valid
              if not (i < num_particles and j < num_particles and i < len(all_patch_data) and j < len(all_patch_data) and \
                  p_idx_i < len(all_patch_data[i]) and p_idx_j < len(all_patch_data[j])):
                   bonds_to_break.append(bond_key); continue

              patch_i_pos = all_patch_data[i][p_idx_i]["position"]
              patch_j_pos = all_patch_data[j][p_idx_j]["position"]
              r_patch_raw = patch_i_pos - patch_j_pos
              r_patch_pbc = r_patch_raw - box_size * np.round(r_patch_raw / box_size)
              if np.linalg.norm(r_patch_pbc) > bond_break_distance:
                   bonds_to_break.append(bond_key)
         # else: # Center-center bonds are not formed by default in this configuration, but can be added
             # i,j = bond_key
             # if not (i < num_particles and j < num_particles): bonds_to_break.append(bond_key); continue
             # r_cc_raw = positions[j,:] - positions[i,:]
             # r_cc_pbc = r_cc_raw - box_size * np.round(r_cc_raw / box_size)
             # if np.linalg.norm(r_cc_pbc) > bonding_params.get("bond_break_distance", 2.0): # Use specific param for CC if needed
             #      bonds_to_break.append(bond_key)

    for bond_key in bonds_to_break:
        if bond_key in updated_bonds: del updated_bonds[bond_key]


    # --- Bond Formation (Patch-Based) ---
    if patch_enabled and patch_definitions:
        # Determine search cutoff based on potential range or break distance
        formation_search_cutoff = bond_break_distance # Max distance to consider formation
        max_patch_extension = 0
        for p_type, p_defs in patch_definitions.items():
            for p_spec in p_defs: max_patch_extension = max(max_patch_extension, p_spec.get("distance",0))


        for i in range(num_particles):
            for j in range(i + 1, num_particles):
                if not all_patch_data[i] or not all_patch_data[j]: continue

                r_ij_center_vec = positions[i, :] - positions[j, :]
                r_ij_center_vec_pbc = r_ij_center_vec - box_size * np.round(r_ij_center_vec / box_size)
                if np.linalg.norm(r_ij_center_vec_pbc) > formation_search_cutoff + 2 * max_patch_extension: continue


                for p_idx_i, patch_i_data in enumerate(all_patch_data[i]):
                    for p_idx_j, patch_j_data in enumerate(all_patch_data[j]):
                        potential_bond_key = tuple(sorted(((i, p_idx_i), (j, p_idx_j))))
                        if potential_bond_key in updated_bonds: continue

                        patch_i_pos, patch_j_pos = patch_i_data["position"], patch_j_data["position"]
                        patch_i_type, patch_j_type = patch_i_data["patch_type"], patch_j_data["patch_type"]

                        # Type compatibility check
                        if not (0 <= patch_i_type < patch_type_compatibility_matrix.shape[0] and \
                                0 <= patch_j_type < patch_type_compatibility_matrix.shape[1] and \
                                patch_type_compatibility_matrix[patch_i_type, patch_j_type]):
                             continue

                        r_patch_raw = patch_i_pos - patch_j_pos
                        r_patch_pbc = r_patch_raw - box_size * np.round(r_patch_raw / box_size)
                        r_patch_mag = np.linalg.norm(r_patch_pbc)

                        # Distance criteria for formation
                        # Use characteristic distance from bond potential type
                        char_bond_dist = bonding_params.get("patch_bond_potential", {}).get("harmonic", {}).get("bond_distance", bond_break_distance)
                        if bonding_params.get("patch_bond_potential",{}).get("type") == "lennard_jones":
                            char_bond_dist = bonding_params.get("patch_bond_potential",{}).get("lennard_jones",{}).get("sigma", bond_break_distance)
                        elif bonding_params.get("patch_bond_potential",{}).get("type") == "square_well":
                            char_bond_dist = bonding_params.get("patch_bond_potential",{}).get("square_well",{}).get("sigma", bond_break_distance)


                        form_dist_met = False
                        if r_patch_mag < bond_break_distance: # Must be within break distance
                             if abs(r_patch_mag - char_bond_dist) < distance_tolerance : form_dist_met = True
                             if bonding_params.get("patch_bond_potential",{}).get("type") == "square_well":
                                 sw_sig = bonding_params.get("patch_bond_potential",{}).get("square_well",{}).get("sigma", bond_break_distance)
                                 sw_lam = bonding_params.get("patch_bond_potential",{}).get("square_well",{}).get("lambda", 1.0)
                                 if sw_sig <= r_patch_mag < sw_sig * sw_lam : form_dist_met = True


                        if form_dist_met:
                            # Orientation alignment (if enabled)
                            orient_align_met = True
                            if orientation_alignment_tolerance is not None and orientation_alignment_tolerance < np.pi: # Check if non-trivial tolerance
                                ideal_angle_map = particle_parameters["forces"]["orientation_potential"]["bond_angle_potential"].get("ideal_angle_mapping", {})

                                # Patch i orientation
                                particle_type_i_int = types[i]
                                # Defensive initialization for pdefs_i
                                pdefs_i = []
                                if particle_type_i_int in patch_definitions:
                                     pdefs_i = patch_definitions.get(particle_type_i_int)
                                elif str(particle_type_i_int) in patch_definitions: # Fallback to string key if int key not found
                                     pdefs_i = patch_definitions.get(str(particle_type_i_int))
                                # Ensure it is a list
                                if not isinstance(pdefs_i, list): pdefs_i = []

                                if p_idx_i >= len(pdefs_i): continue
                                angle_rel_i = pdefs_i[p_idx_i].get("angle_relative_to_particle",0.0)
                                abs_angle_i = orientations[i] + angle_rel_i
                                patch_i_orient_vec = np.array([np.cos(abs_angle_i), np.sin(abs_angle_i)])
                                ideal_i = ideal_angle_map.get(patch_i_type, 0.0) # ideal_angle_map has int keys
                                angle_i_to_bond = np.arctan2(patch_i_orient_vec[0]*(r_patch_pbc[1]/r_patch_mag) - patch_i_orient_vec[1]*(r_patch_pbc[0]/r_patch_mag), np.dot(patch_i_orient_vec, r_patch_pbc/r_patch_mag))
                                dev_i = abs(np.mod(angle_i_to_bond - ideal_i + np.pi, 2*np.pi) - np.pi)


                                # Patch j orientation
                                particle_type_j_int = types[j]
                                # Defensive initialization for pdefs_j
                                pdefs_j = []
                                if particle_type_j_int in patch_definitions:
                                     pdefs_j = patch_definitions.get(particle_type_j_int)
                                elif str(particle_type_j_int) in patch_definitions: # Fallback to string key
                                     pdefs_j = patch_definitions.get(str(particle_type_j_int))
                                # Ensure it is a list
                                if not isinstance(pdefs_j, list): pdefs_j = []

                                if p_idx_j >= len(pdefs_j): continue
                                angle_rel_j = pdefs_j[p_idx_j].get("angle_relative_to_particle",0.0)
                                abs_angle_j = orientations[j] + angle_rel_j
                                patch_j_orient_vec = np.array([np.cos(abs_angle_j), np.sin(abs_angle_j)])
                                ideal_j = ideal_angle_map.get(patch_j_type, 0.0)
                                angle_j_to_bond = np.arctan2(patch_j_orient_vec[0]*(-bond_vec_dir[1])-patch_j_orient_vec[1]*(-bond_vec_dir[0]), np.dot(patch_j_orient_vec,-bond_vec_dir))
                                dev_j = abs(np.mod(angle_j_to_bond - ideal_j + np.pi, 2*np.pi) - np.pi)


                                if dev_i > orientation_alignment_tolerance or dev_j > orientation_alignment_tolerance:
                                     orient_align_met = False

                            if orient_align_met:
                                 # Form bond
                                 initial_strength = bonding_params.get("patch_bond_potential",{}).get("harmonic",{}).get("bond_strength", 100.0) # Store initial harmonic for adaptive
                                 updated_bonds[potential_bond_key] = {'formed_step': current_step, 'initial_strength': initial_strength, 'patch_pair': (p_idx_i, p_idx_j)}

    return updated_bonds


# Function for Particle Creation (remains the same, uses particle_parameters)
def create_particles(positions, velocities, orientations, types, masses, bonds, particle_parameters, current_step, all_patch_data):
    """
    Handles particle creation based on defined criteria.
    (Placeholder - actual creation logic needs to be implemented based on triggers,
    possibly influenced by brain state).
    Uses particle_parameters for creation criteria and new particle properties.
    """
    creation_params = particle_parameters.get("particle_creation", {})
    if not creation_params.get("enabled", False) or random.random() > creation_params.get("creation_rate",0.0): # Stochastic rate
        return positions, velocities, orientations, types, masses, bonds, all_patch_data, len(positions)

    num_existing_particles = len(positions)
    new_particle_config = creation_params.get("new_particle", {})
    trigger_config = creation_params.get("trigger", {})
    # Placeholder for trigger logic (e.g. a specific particle type initiating creation)
    # For simplicity, let's try to add one particle if any particle exists
    if num_existing_particles == 0:
        return positions, velocities, orientations, types, masses, bonds, all_patch_data, num_existing_particles


    # Select a random existing particle as parent (very simplified trigger)
    parent_idx = random.randrange(num_existing_particles)
    parent_pos = positions[parent_idx]
    parent_vel = velocities[parent_idx]
    parent_orient = orientations[parent_idx]
    # parent_ang_vel = angular_velocities[parent_idx] # If angular_velocities available


    new_type = new_particle_config.get("type", 0)
    new_pos_offset = np.array([ (np.random.rand()-0.5)*2 , (np.random.rand()-0.5)*2 ]) # Random offset
    new_pos = parent_pos + new_pos_offset
    new_vel = parent_vel * new_particle_config.get("initial_velocity_scale", 0.1)

    new_orient_config = new_particle_config.get("angular_initialization", particle_parameters["initial_conditions"]["new_particle_angular_initialization"])
    new_orient = parent_orient if new_orient_config["orientation_type"] == "copy_parent" else new_orient_config["orientation_angle"]
    # new_ang_vel = parent_ang_vel * new_orient_config["angular_velocity_scale"] if new_orient_config["angular_velocity_type"] == "copy_scaled_parent" else 0.0


    new_mass = particle_parameters["initial_conditions"]["mass_mapping"].get(new_type, 1.0)

    # Append new particle
    positions = np.vstack([positions, new_pos])
    velocities = np.vstack([velocities, new_vel])
    orientations = np.append(orientations, new_orient)
    types = np.append(types, new_type)
    masses = np.append(masses, new_mass)
    # angular_velocities = np.append(angular_velocities, new_ang_vel) # If tracking
    # accelerations & angular_accelerations will need to be resized too

    # Update all_patch_data: add empty list for the new particle
    all_patch_data.append([]) # This will be properly populated in next force calculation

    num_particles = len(positions)
    # print(f"Particle created. Total particles: {num_particles}")
    return positions, velocities, orientations, types, masses, bonds, all_patch_data, num_particles


# Function for Particle Deletion (remains the same, uses particle_parameters)
def delete_particles(positions, velocities, orientations, types, masses, bonds, particle_parameters, current_step, all_patch_data):
    """
    Handles particle deletion based on defined criteria.
    (Placeholder - actual deletion logic needs to be implemented based on triggers,
    possibly influenced by brain state).
    Uses particle_parameters for deletion criteria.
    """
    deletion_params = particle_parameters.get("particle_deletion", {})
    if not deletion_params.get("enabled", False) or random.random() > deletion_params.get("deletion_rate",0.0):
        return positions, velocities, orientations, types, masses, bonds, all_patch_data, len(positions)

    num_particles_before_del = len(positions)
    if num_particles_before_del == 0: return positions, velocities, orientations, types, masses, bonds, all_patch_data, 0


    trigger = deletion_params.get("trigger", {})
    condition = trigger.get("condition", None)
    buffer_distance = trigger.get("buffer_distance", 0.0)
    particles_to_delete_indices = []

    if condition == "out_of_bounds":
         boundaries = particle_parameters["boundaries"]
         x_min, x_max = boundaries["x_min"], boundaries["x_max"]
         y_min, y_max = boundaries["y_min"], boundaries["y_max"]
         for i in range(num_particles_before_del):
              if (positions[i, 0] < x_min - buffer_distance or positions[i, 0] > x_max + buffer_distance or
                  positions[i, 1] < y_min - buffer_distance or positions[i, 1] > y_max + buffer_distance):
                   particles_to_delete_indices.append(i)
    # Add other deletion conditions if needed

    if particles_to_delete_indices:
         keep_mask = np.ones(num_particles_before_del, dtype=bool)
         keep_mask[particles_to_delete_indices] = False

         positions = positions[keep_mask]
         velocities = velocities[keep_mask]
         orientations = orientations[keep_mask]
         types = types[keep_mask]
         masses = masses[keep_mask]
         # Resize accelerations if they are managed outside:
         # accelerations = accelerations[keep_mask]
         # angular_velocities = angular_velocities[keep_mask]
         # angular_accelerations = angular_accelerations[keep_mask]

         all_patch_data_updated = [patch_list for i, patch_list in enumerate(all_patch_data) if keep_mask[i]]
         all_patch_data = all_patch_data_updated

         # Update bonds: remove bonds involving deleted particles and re-index remaining
         old_to_new_map = {old_idx: new_idx for new_idx, old_idx in enumerate(np.where(keep_mask)[0])}
         updated_bonds = {}
         for bond_key, bond_info in bonds.items():
             is_patch_bond = isinstance(bond_key, tuple) and len(bond_key) == 2 and isinstance(bond_key[0], tuple)
             if is_patch_bond:
                  (i, p_idx_i), (j, p_idx_j) = bond_key
                  if i in old_to_new_map and j in old_to_new_map:
                       updated_bonds[tuple(sorted(((old_to_new_map[i], p_idx_i), (old_to_new_map[j], p_idx_j))))] = bond_info
             # else: # Center-center bond
                 # i,j = bond_key
                 # if i in old_to_new_map and j in old_to_new_map:
                 #      updated_bonds[tuple(sorted((old_to_new_map[i], old_to_new_map[j])))] = bond_info
         bonds = updated_bonds
         # print(f"Deleted {len(particles_to_delete_indices)} particles. Remaining: {len(positions)}")


    return positions, velocities, orientations, types, masses, bonds, all_patch_data, len(positions)


# Function for State Change on Bond Formation (remains the same, uses particle_parameters)
def apply_state_change_on_bond_form(types, bonds, particle_parameters, current_step):
    """
    Changes the type of a particle when a specific bond is formed.
    (Placeholder - actual logic needs to be implemented based on triggers).
    Uses particle_parameters for state change criteria.
    """
    state_change_params = particle_parameters.get("state_change", {})
    if not state_change_params.get("enabled", False):
        return types

    on_bond_form_params = state_change_params.get("on_bond_form", {})
    from_type_config = on_bond_form_params.get("from_type", None) # This could be an int
    to_type_config = on_bond_form_params.get("to_type", None) # This could be an int

    if from_type_config is None or to_type_config is None:
         return types

    updated_types = np.copy(types)
    num_particles = updated_types.shape[0]

    for bond_key, bond_info in bonds.items():
         if bond_info.get('formed_step', -1) == current_step: # Bond formed in current step
              is_patch_bond = isinstance(bond_key, tuple) and len(bond_key) == 2 and isinstance(bond_key[0], tuple)
              if is_patch_bond:
                   (i, _), (j, _) = bond_key # Don't need patch indices for type change of particle
                   if i < num_particles and updated_types[i] == from_type_config: updated_types[i] = to_type_config
                   if j < num_particles and updated_types[j] == from_type_config: updated_types[j] = to_type_config
              # else: # Center-center bond
                   # i,j = bond_key
                   # if i < num_particles and updated_types[i] == from_type_config : updated_types[i] = to_type_config
                   # if j < num_particles and updated_types[j] == from_type_config : updated_types[j] = to_type_config
    return updated_types


# Function to Get Cluster Labels (remains the same)
def get_cluster_labels_for_frame(positions, bonds, num_particles):
    """
    Identifies clusters of particles based on active bonds.
    """
    if num_particles == 0:
        return np.array([], dtype=int)

    graph = scipy.sparse.lil_matrix((num_particles, num_particles), dtype=int)
    for bond_key in bonds.keys():
         is_patch_bond = isinstance(bond_key, tuple) and len(bond_key) == 2 and isinstance(bond_key[0], tuple)
         if is_patch_bond:
              (i, _), (j, _) = bond_key # Particle indices are i and j
              if i < num_particles and j < num_particles:
                   graph[i, j] = 1; graph[j, i] = 1
         # else: # Center-center bond
             # i, j = bond_key
             # if i < num_particles and j < num_particles:
             #      graph[i, j] = 1; graph[j, i] = 1

    n_components, labels = scipy.sparse.csgraph.connected_components(graph, directed=False, connection='weak') # Weak for undirected
    return labels


# Function to Save Simulation State (remains the same, uses particle_parameters)
def save_simulation_state(step, positions, velocities, orientations, types, masses, bonds, particle_parameters):
    """
    Saves the current state of the particle simulation to files.
    Uses particle_parameters for the save directory.
    """
    save_directory = particle_parameters["saving"]["save_directory"]
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Convert bonds to JSON-serializable format (list of lists for keys, info as dict)
    serializable_bonds = {}
    for k, v in bonds.items():
        # Key: Convert ((p1_idx, patch1_idx), (p2_idx, patch2_idx)) to string "p1_patch1-p2_patch2"
        # Or handle simpler int-tuple keys for center-center bonds.
        # For pickle, tuples are fine as keys. If this was for JSON, more care needed.
        # Since it's pickle, we can store bonds dictionary directly.
        serializable_bonds[k] = v


    state = {
        "step": step,
        "positions": positions.tolist(),
        "velocities": velocities.tolist(),
        "orientations": orientations.tolist(),
        "types": types.tolist(),
        "masses": masses.tolist(),
        "bonds": serializable_bonds, # Store the dict as-is for pickle
        "parameters": particle_parameters
    }

    filepath = os.path.join(save_directory, f"particle_state_step_{step:06d}.pkl")
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    except Exception as e:
        print(f"Error saving particle simulation state to {filepath}: {e}")


# Function to Load Simulation State (remains the same, uses particle_parameters)
def load_simulation_state(particle_parameters_input): # Rename to avoid conflict with global
    """
    Loads the latest particle simulation state from files in the load directory.
    Uses particle_parameters_input for the load directory.
    """
    load_directory = particle_parameters_input["saving"]["load_directory"]
    if not os.path.exists(load_directory):
        print(f"Particle load directory '{load_directory}' not found. Starting from initial conditions.")
        return None

    state_files = [f for f in os.listdir(load_directory) if f.startswith("particle_state_step_") and f.endswith(".pkl")]
    if not state_files:
        print(f"No particle simulation state files found in '{load_directory}'. Starting from initial conditions.")
        return None

    try: # Add try block for parsing step numbers
        latest_step = max([int(f.split('_')[-1].split('.')[0]) for f in state_files])
    except ValueError:
        print(f"Error parsing step numbers from filenames in '{load_directory}'. Starting fresh.")
        return None

    latest_filepath = os.path.join(load_directory, f"particle_state_step_{latest_step:06d}.pkl")

    try:
        with open(latest_filepath, 'rb') as f:
            state = pickle.load(f)
        print(f"Particle simulation state loaded successfully from '{latest_filepath}'")

        # Use parameters from the loaded state primarily
        loaded_parameters = state.get("parameters", particle_parameters_input)
        # Override with current script's visualization params if desired, or merge carefully
        # For now, strictly use loaded_parameters for consistency of the loaded state.

        loaded_types_np = np.array(state["types"], dtype=int)
        vis_params = loaded_parameters.get("visualization", {})
        patches_vis_params = vis_params.get("patches", {})
        color_mapping_from_loaded = patches_vis_params.get("color_mapping", {})
        # Ensure color_mapping keys are int for lookup, as types are int
        color_mapping_int_keys = {int(k):v for k,v in color_mapping_from_loaded.items()} if isinstance(color_mapping_from_loaded,dict) else {}


        loaded_colors = [color_mapping_int_keys.get(t, 'gray') for t in loaded_types_np]


        # Accelerations and angular velocities/accelerations are re-calculated at simulation start.
        num_loaded_particles = len(state["positions"])
        return (
            state["step"],
            np.array(state["positions"]),
            np.array(state["velocities"]),
            np.zeros((num_loaded_particles,2)) if num_loaded_particles > 0 else np.array([]).reshape(0,2), # Accelerations placeholder
            state.get("bonds", {}), # Bonds directly from pickle
            loaded_types_np,
            np.array(state["masses"]),
            loaded_colors,
            np.array(state["orientations"]),
            np.zeros(num_loaded_particles) if num_loaded_particles > 0 else np.array([]), # Angular velocities placeholder
            np.zeros(num_loaded_particles) if num_loaded_particles > 0 else np.array([]), # Angular accelerations placeholder
            num_loaded_particles,
            loaded_parameters, # Return the parameters that were part of the loaded state
            [] # Placeholder for all_patch_data, regenerated in first step
        )

    except FileNotFoundError:
        print(f"Error loading particle state from '{latest_filepath}'. File not found.")
        return None
    except Exception as e:
        print(f"An error occurred loading particle simulation state: {e}. Starting from initial conditions.")
        return None


# Analysis Function: Radial Distribution Function (RDF) (remains the same, uses particle_parameters)
def calculate_radial_distribution_function(positions_history, particle_parameters):
    """
    Calculates the radial distribution function (g(r)) for the particle system.
    Calculates g(r) averaged over specified frames.
    Uses particle_parameters for analysis settings and boundaries.
    Requires positions_history.
    """
    analysis_params = particle_parameters["analysis"]
    rdf_dr = analysis_params.get("rdf_dr", 0.1)
    boundaries = particle_parameters["boundaries"] # Get boundaries from particle_parameters
    rdf_rmax = analysis_params.get("rdf_rmax", (boundaries["x_max"] - boundaries["x_min"]) / 2.0)
    rdf_start_frame = analysis_params.get("rdf_start_frame", 0)


    if not positions_history or len(positions_history) <= rdf_start_frame:
        # print("Warning: No particle position data available for RDF or start frame out of bounds.")
        return np.array([]), np.array([])


    box_size = np.array([boundaries["x_max"] - boundaries["x_min"], boundaries["y_max"] - boundaries["y_min"]])
    if box_size[0] <=0 or box_size[1] <=0 : return np.array([]), np.array([]) # Invalid box


    num_bins = int(rdf_rmax / rdf_dr)
    if num_bins <=0 : return np.array([]), np.array([])
    distance_counts = np.zeros(num_bins)
    bin_edges = np.linspace(0, rdf_rmax, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total_number_density_sum = 0.0 # Sum of number densities over frames
    num_frames_averaged = 0
    total_particles_in_avg_frames = 0 # Sum of N over averaged frames


    for frame_index in range(rdf_start_frame, len(positions_history)):
        positions = positions_history[frame_index]
        if not isinstance(positions, np.ndarray) or positions.ndim != 2 or positions.shape[1]!=2: continue # Invalid position data for frame
        num_particles_in_frame = positions.shape[0]
        if num_particles_in_frame < 2: continue

        num_frames_averaged += 1
        total_particles_in_avg_frames += num_particles_in_frame


        r_vec_all_pairs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        r_vec_all_pairs_pbc = r_vec_all_pairs - box_size * np.round(r_vec_all_pairs / box_size)
        r_mag_all_pairs = np.sqrt(np.sum(r_vec_all_pairs_pbc**2, axis=2))

        upper_triangle_indices = np.triu_indices_from(r_mag_all_pairs, k=1)
        distances = r_mag_all_pairs[upper_triangle_indices]
        distances_within_rmax = distances[distances < rdf_rmax]

        valid_distances = distances_within_rmax[np.isfinite(distances_within_rmax)]
        if valid_distances.size > 0:
             counts, _ = np.histogram(valid_distances, bins=bin_edges)
             distance_counts += counts


        area = box_size[0] * box_size[1]
        if area > DEFAULT_EPSILON: # Ensure area is positive
             total_number_density_sum += num_particles_in_frame / area


    if num_frames_averaged == 0 or np.sum(distance_counts) == 0:
         # print("Warning: No valid frames or pairs found for RDF calculation.")
         return np.array([]), np.array([])

    avg_num_particles_for_norm = total_particles_in_avg_frames / num_frames_averaged
    avg_number_density_for_norm = total_number_density_sum / num_frames_averaged


    rdf_g_r = np.zeros(num_bins)
    for i in range(num_bins):
        r_inner, r_outer = bin_edges[i], bin_edges[i+1]
        area_of_bin = np.pi * (r_outer**2 - r_inner**2)
        # Expected number of pairs in this bin FOR ONE FRAME, for N particles, with density rho:
        # N * rho * area_bin / 2. We sum counts over all frames, so multiply by num_frames_averaged.
        expected_pairs_in_bin_total = num_frames_averaged * avg_num_particles_for_norm * avg_number_density_for_norm * area_of_bin / 2.0

        if expected_pairs_in_bin_total > DEFAULT_EPSILON:
             rdf_g_r[i] = distance_counts[i] / expected_pairs_in_bin_total
        else: rdf_g_r[i] = 0.0

    return bin_centers, rdf_g_r


# Plotting Function for RDF (remains the same, uses particle_parameters)
def plot_rdf(rdf_bin_centers, rdf_g_r, particle_parameters):
    if rdf_bin_centers.size == 0 or rdf_g_r.size == 0: return # Check if arrays are empty
    save_directory = particle_parameters["saving"]["save_directory"]
    if not os.path.exists(save_directory): os.makedirs(save_directory)
    plt.figure(); plt.plot(rdf_bin_centers, rdf_g_r); plt.xlabel("Distance (r)")
    plt.ylabel("g(r)"); plt.title("Particle Radial Distribution Function"); plt.grid(True)
    ymin = 0.0; ymax = max(2.0, np.max(rdf_g_r[np.isfinite(rdf_g_r)]) * 1.1 if np.any(np.isfinite(rdf_g_r)) else 2.0)
    plt.ylim(ymin, ymax)
    plt.savefig(os.path.join(save_directory, "particle_rdf_plot.png")); plt.close()
    print(f"Particle RDF plot saved to {os.path.join(save_directory, 'particle_rdf_plot.png')}")


# Analysis Function: Mean Squared Displacement (MSD) (remains the same, uses particle_parameters)
def calculate_mean_squared_displacement(positions_history, final_types, particle_parameters):
    analysis_params = particle_parameters["analysis"]
    msd_start_frame = analysis_params.get("msd_start_frame", analysis_params.get("rdf_start_frame", 0))

    if not positions_history or len(positions_history) <= msd_start_frame +1: # Need at least 2 frames from start
        # print("Warning: Insufficient position data for MSD.")
        return np.array([]), np.array([]), {}

    num_total_frames = len(positions_history)
    boundaries = particle_parameters["boundaries"]
    box_size = np.array([boundaries["x_max"] - boundaries["x_min"], boundaries["y_max"] - boundaries["y_min"]])
    dt_sim = particle_parameters["simulation"]["dt"]


    # Time lags to calculate MSD for (up to half the duration of analyzed trajectory segment)
    max_lag_steps = (num_total_frames - msd_start_frame) // 2
    if max_lag_steps <=0: return np.array([]), np.array([]), {} # Not enough steps for any lag
    msd_time_lags_dt = np.arange(1, max_lag_steps + 1) # Time lags in units of steps
    msd_time_points_sec = msd_time_lags_dt * dt_sim


    overall_msd = np.zeros(max_lag_steps)
    msd_per_type = collections.defaultdict(lambda: np.zeros(max_lag_steps))
    # Counts for averaging, ensure final_types is an np.array
    final_types_np = np.array(final_types) if not isinstance(final_types, np.ndarray) else final_types


    for lag_idx, time_lag_steps in enumerate(msd_time_lags_dt):
        squared_displacements_for_lag = []
        type_specific_sq_disp_for_lag = collections.defaultdict(list)

        # Iterate over possible start times (origins) for this lag
        for origin_frame_idx in range(msd_start_frame, num_total_frames - time_lag_steps):
            pos_origin = positions_history[origin_frame_idx]
            pos_lagged = positions_history[origin_frame_idx + time_lag_steps]

            # Ensure particle count consistency for this specific pair of frames
            if pos_origin.shape[0] != pos_lagged.shape[0] or pos_origin.shape[0] == 0:
                continue # Skip if particle numbers differ or no particles

            # Assume final_types corresponds to particles at origin_frame_idx
            # This is an approximation if particles are created/deleted
            current_types_for_origin = final_types_np
            if len(final_types_np) != pos_origin.shape[0]:
                 # If final_types doesn't match, try to get types from types_history (if available and passed)
                 # For simplicity here, we proceed with warning or skip type-specific
                 # print(f"Warning: final_types length mismatch for MSD at frame {origin_frame_idx}. Type-specific MSD may be affected.")
                 # If types_history was passed:
                 # current_types_for_origin = types_history[origin_frame_idx] if origin_frame_idx < len(types_history) else final_types_np
                 pass


            disp_raw = pos_lagged - pos_origin
            disp_pbc = disp_raw - box_size * np.round(disp_raw / box_size)
            sq_disp_per_particle = np.sum(disp_pbc**2, axis=1)

            squared_displacements_for_lag.extend(sq_disp_per_particle)
            if len(current_types_for_origin) == len(sq_disp_per_particle): # If types array matches
                 for p_idx, sq_d in enumerate(sq_disp_per_particle):
                      type_specific_sq_disp_for_lag[current_types_for_origin[p_idx]].append(sq_d)


        if squared_displacements_for_lag:
            overall_msd[lag_idx] = np.mean(squared_displacements_for_lag)
        for p_type, disp_list in type_specific_sq_disp_for_lag.items():
            if disp_list: msd_per_type[p_type][lag_idx] = np.mean(disp_list)

    return msd_time_points_sec, overall_msd, dict(msd_per_type) # Convert defaultdict to dict


# Plotting Function for MSD (remains the same, uses particle_parameters)
def plot_msd(msd_time_points, overall_msd, type_msd_dict, particle_parameters):
    if msd_time_points.size == 0 or overall_msd.size == 0: return
    save_directory = particle_parameters["saving"]["save_directory"]
    if not os.path.exists(save_directory): os.makedirs(save_directory)

    plt.figure(); plt.plot(msd_time_points, overall_msd, label="Overall MSD")
    color_map_vis = particle_parameters.get("visualization", {}).get("patches", {}).get("color_mapping", {})
    # Ensure color_map_vis keys are int if types are int
    color_map_int_keys = {int(k):v for k,v in color_map_vis.items()} if isinstance(color_map_vis,dict) else {}


    for p_type, msd_arr in type_msd_dict.items():
        if msd_arr.size == msd_time_points.size:
            plt.plot(msd_time_points, msd_arr, label=f"Type {p_type} MSD", color=color_map_int_keys.get(p_type, 'gray'))
    plt.xlabel("Time (s)"); plt.ylabel("MSD (distance$^2$)")
    plt.title("Particle Mean Squared Displacement"); plt.legend(); plt.grid(True); plt.xscale('log'); plt.yscale('log') # Often plotted log-log
    plt.savefig(os.path.join(save_directory, "particle_msd_plot.png")); plt.close()
    print(f"Particle MSD plot saved to {os.path.join(save_directory, 'particle_msd_plot.png')}")


# Analysis Function: Bond Angle Distribution (remains the same, uses particle_parameters)
def calculate_bond_angle_distribution(positions_history, orientations_history, all_patch_data_history, bonds_history, types_history, particle_parameters):
    orient_pot_params = particle_parameters["forces"].get("orientation_potential", {})
    bond_angle_pot_params = orient_pot_params.get("bond_angle_potential", {})
    if not bond_angle_pot_params.get("enabled", False): return np.array([]), np.array([])

    patch_defs = particle_parameters["forces"].get("patch_params", {}).get("patch_definitions", {})
    if not patch_defs: return np.array([]), np.array([])

    ideal_angle_map = bond_angle_pot_params.get("ideal_angle_mapping", {}) # Keys are int

    angles_dev = []
    boundaries = particle_parameters["boundaries"]
    box_size = np.array([boundaries["x_max"] - boundaries["x_min"], boundaries["y_max"] - boundaries["y_min"]])

    min_history_len = min(len(h) for h in [positions_history, orientations_history, all_patch_data_history, bonds_history, types_history])

    for frame_idx in range(min_history_len):
        bonds_f, pos_f, orient_f, patches_f, types_f = bonds_history[frame_idx], positions_history[frame_idx], orientations_history[frame_idx], all_patch_data_history[frame_idx], types_history[frame_idx]
        num_p_f = pos_f.shape[0]

        for bond_key in bonds_f.keys():
            if not (isinstance(bond_key, tuple) and len(bond_key) == 2 and isinstance(bond_key[0], tuple)): continue
            (i, p_idx_i), (j, p_idx_j) = bond_key
            if not (i<num_p_f and j<num_p_f and i<len(patches_f) and j<len(patches_f) and p_idx_i<len(patches_f[i]) and p_idx_j<len(patches_f[j])): continue

            patch_i_dat, patch_j_dat = patches_f[i][p_idx_i], patches_f[j][p_idx_j]
            patch_i_type, patch_j_type = patch_i_dat["patch_type"], patch_j_dat["patch_type"]

            r_patch_raw = patch_j_dat["position"] - patch_i_dat["position"]
            r_patch_pbc = r_patch_raw - box_size * np.round(r_patch_raw / box_size)
            r_patch_mag = np.linalg.norm(r_patch_pbc)
            if r_patch_mag < DEFAULT_EPSILON: continue
            bond_vec_dir = r_patch_pbc / r_patch_mag

            particle_type_i_int = types_f[i]
            # Defensive initialization for pdefs_i
            pdefs_i = []
            if particle_type_i_int in patch_defs:
                pdefs_i = patch_defs.get(particle_type_i_int)
            elif str(particle_type_i_int) in patch_defs:
                pdefs_i = patch_defs.get(str(particle_type_i_int))
            if not isinstance(pdefs_i, list): pdefs_i = [] # Final safeguard

            if p_idx_i >= len(pdefs_i): continue
            angle_rel_i = pdefs_i[p_idx_i].get("angle_relative_to_particle",0.0)
            abs_angle_i = orient_f[i] + angle_rel_i
            patch_i_orient_vec = np.array([np.cos(abs_angle_i), np.sin(abs_angle_i)])
            ideal_i_rad = ideal_angle_map.get(patch_i_type, 0.0) # patch_i_type is int
            actual_i_angle = np.arctan2(patch_i_orient_vec[0]*bond_vec_dir[1]-patch_i_orient_vec[1]*bond_vec_dir[0], np.dot(patch_i_orient_vec,bond_vec_dir))
            angles_dev.append(abs(np.mod(actual_i_angle - ideal_i_rad + np.pi, 2*np.pi) - np.pi))


            particle_type_j_int = types_f[j]
            # Defensive initialization for pdefs_j
            pdefs_j = []
            if particle_type_j_int in patch_defs:
                pdefs_j = patch_defs.get(particle_type_j_int)
            elif str(particle_type_j_int) in patch_defs:
                pdefs_j = patch_defs.get(str(particle_type_j_int))
            if not isinstance(pdefs_j, list): pdefs_j = [] # Final safeguard

            if p_idx_j >= len(pdefs_j): continue
            angle_rel_j = pdefs_j[p_idx_j].get("angle_relative_to_particle",0.0)
            abs_angle_j = orient_f[j] + angle_rel_j
            patch_j_orient_vec = np.array([np.cos(abs_angle_j), np.sin(abs_angle_j)])
            ideal_j_rad = ideal_angle_map.get(patch_j_type, 0.0)
            actual_j_angle = np.arctan2(patch_j_orient_vec[0]*(-bond_vec_dir[1])-patch_j_orient_vec[1]*(-bond_vec_dir[0]), np.dot(patch_j_orient_vec,-bond_vec_dir))
            angles_dev.append(abs(np.mod(actual_j_angle - ideal_j_rad + np.pi, 2*np.pi) - np.pi))

    if not angles_dev: return np.array([]), np.array([])
    hist, edges = np.histogram(angles_dev, bins=50, range=(0,np.pi), density=True)
    centers = (edges[:-1] + edges[1:])/2
    return centers, hist


# Plotting Function for Bond Angle Distribution (remains the same, uses particle_parameters)
def plot_bond_angle_distribution(bin_centers, angle_counts, particle_parameters):
    if bin_centers.size == 0 or angle_counts.size == 0: return
    save_dir = particle_parameters["saving"]["save_directory"]
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    plt.figure(); plt.bar(bin_centers, angle_counts, width=(bin_centers[1]-bin_centers[0] if len(bin_centers)>1 else 0.1) , edgecolor='black')
    plt.xlabel("Angle Deviation (radians)"); plt.ylabel("Probability Density")
    plt.title("Patch Bond Angle Distribution"); plt.grid(axis='y')
    plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], ['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
    plt.savefig(os.path.join(save_dir, "particle_bond_angle_distribution.png")); plt.close()
    print(f"Particle bond angle distribution plot saved to {os.path.join(save_dir, 'particle_bond_angle_distribution.png')}")


# Analysis Function: Cluster Size Distribution (remains the same, uses particle_parameters)
def calculate_cluster_size_distribution(bonds_history, positions_history, particle_parameters):
    if not bonds_history or not positions_history : return np.array([]), np.array([])
    all_sizes = []
    min_len = min(len(bonds_history), len(positions_history))
    for frame_idx in range(min_len):
        bonds_f, pos_f = bonds_history[frame_idx], positions_history[frame_idx]
        num_p = pos_f.shape[0]
        if num_p > 0:
            labels = get_cluster_labels_for_frame(pos_f, bonds_f, num_p)
            _, counts = np.unique(labels, return_counts=True)
            all_sizes.extend(counts)
    if not all_sizes: return np.array([]), np.array([])
    max_size = max(all_sizes) if all_sizes else 1
    hist, edges = np.histogram(all_sizes, bins=np.arange(0.5, max_size+1.5,1.0), density=True)
    centers = np.arange(1, max_size+1)
    # Ensure hist matches centers length if max_size was small
    if len(hist) < len(centers) and len(centers) == 1 and max_size == 1: # common for single particles
        hist = np.array([1.0]) if sum(all_sizes)==len(all_sizes) and all_sizes[0]==1 else np.array([0.0])


    return centers, hist


# Plotting Function for Cluster Size Distribution (remains the same, uses particle_parameters)
def plot_cluster_size_distribution(cluster_sizes, cluster_counts, particle_parameters):
    if cluster_sizes.size == 0 or cluster_counts.size == 0: return
    save_dir = particle_parameters["saving"]["save_directory"]
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    plt.figure(); plt.bar(cluster_sizes, cluster_counts, width=1.0, edgecolor='black')
    plt.xlabel("Cluster Size"); plt.ylabel("Probability"); plt.title("Particle Cluster Size Distribution")
    plt.xticks(cluster_sizes) if len(cluster_sizes) < 20 else plt.xticks(np.arange(min(cluster_sizes),max(cluster_sizes)+1, max(1,int(len(cluster_sizes)/10)) ))

    plt.grid(axis='y'); plt.savefig(os.path.join(save_dir, "particle_cluster_size_distribution.png")); plt.close()
    print(f"Particle cluster size distribution plot saved to {os.path.join(save_dir, 'particle_cluster_size_distribution.png')}")


# Analysis Function: Calculate Nematic Order Parameter (remains the same)
def calculate_nematic_order_parameter(orientations):
    if len(orientations) == 0: return 0.0
    cos_2t, sin_2t = np.cos(2*orientations), np.sin(2*orientations)
    return np.sqrt(np.mean(cos_2t)**2 + np.mean(sin_2t)**2)


# Visualization Function (Adapted for Coupled Simulation)
def visualize_simulation(total_energy_history, positions_history, orientations_history, nematic_order_parameter_history, all_patch_data_history, bonds_history, types_history, particle_parameters):
    save_dir = particle_parameters["saving"]["save_directory"]
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    anim_file = os.path.join(save_dir, particle_parameters["saving"].get("animation_filename", "particle_sim.gif"))
    boundaries = particle_parameters["boundaries"]
    x_min, x_max, y_min, y_max = boundaries["x_min"], boundaries["x_max"], boundaries["y_min"], boundaries["y_max"]

    if total_energy_history:
        plt.figure(); plt.plot(total_energy_history); plt.xlabel("Step"); plt.ylabel("Total Energy")
        plt.title("Particle Total Energy"); plt.grid(True)
        plt.savefig(os.path.join(save_dir, "particle_total_energy_plot.png")); plt.close()
        print(f"Particle total energy plot saved to {os.path.join(save_dir, 'particle_total_energy_plot.png')}")

    if nematic_order_parameter_history:
        plt.figure(); plt.plot(nematic_order_parameter_history); plt.xlabel("Step"); plt.ylabel("Nematic Order (S)")
        plt.title("Particle Nematic Order Parameter"); plt.grid(True); plt.ylim(0, 1.1)
        plt.savefig(os.path.join(save_dir, "particle_nematic_order_parameter_plot.png")); plt.close()
        print(f"Particle nematic order plot saved to {os.path.join(save_dir, 'particle_nematic_order_parameter_plot.png')}")

    if not positions_history: print("No particle positions for animation."); return

    fig_anim, ax_anim = plt.subplots(); ax_anim.set_xlim(x_min, x_max); ax_anim.set_ylim(y_min, y_max)
    ax_anim.set_aspect('equal', adjustable='box'); ax_anim.set_title("Particle Simulation"); ax_anim.grid(True)

    vis_params = particle_parameters["visualization"]
    orient_line_p = vis_params.get("orientation_line", {})
    patches_vis_p = vis_params.get("patches", {})
    bonds_vis_p = vis_params.get("bonds", {})
    clusters_vis_p = vis_params.get("clusters", {})

    particle_scatter = ax_anim.scatter([], [], s=50) # Base size
    max_hist_particles = max(len(p) for p in positions_history if len(p)>0) if any(len(p)>0 for p in positions_history) else 0
    orientation_lines_plt = [ax_anim.plot([], [], color=orient_line_p.get("color", 'k'), lw=orient_line_p.get("linewidth", 1))[0] for _ in range(max_hist_particles)]
    patch_scatter_plt = ax_anim.scatter([], [], s=(patches_vis_p.get("size",0.5)**2)*20, edgecolors=patches_vis_p.get("edgecolor",'k'), zorder=3)
    bond_lines_plt = [] # To store line artists for bonds, cleared each frame
    time_text_plt = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes)

    min_hist_len_anim = min(len(h) for h in [positions_history, orientations_history, all_patch_data_history, bonds_history, types_history])


    def update_anim(frame):
        if frame >= min_hist_len_anim: return [particle_scatter, patch_scatter_plt, time_text_plt] + orientation_lines_plt + bond_lines_plt

        pos_f, orient_f, patches_f_all, bonds_f, types_f = positions_history[frame], orientations_history[frame], all_patch_data_history[frame], bonds_history[frame], types_history[frame]
        num_p_f = pos_f.shape[0]
        if num_p_f == 0: # Handle empty frame
             particle_scatter.set_offsets(np.array([]).reshape(0,2))
             patch_scatter_plt.set_offsets(np.array([]).reshape(0,2))
             for line in orientation_lines_plt: line.set_data([],[])
             for bline in bond_lines_plt: bline.remove()
             bond_lines_plt.clear()
             time_text_plt.set_text(f"Step: {frame}, Time: {frame*particle_parameters['simulation']['dt']:.1f} (No Particles)")
             return [particle_scatter, patch_scatter_plt, time_text_plt] + orientation_lines_plt

        particle_scatter.set_offsets(pos_f)
        # Coloring particles
        current_colors = ['gray'] * num_p_f # Default
        if clusters_vis_p.get("enabled", False):
            labels = get_cluster_labels_for_frame(pos_f, bonds_f, num_p_f)
            unique_labels = np.unique(labels)
            cmap_func = plt.get_cmap(clusters_vis_p.get("colormap", "viridis"))
            current_colors = [cmap_func(l/(len(unique_labels)-1)) if len(unique_labels)>1 else cmap_func(0.5) for l in labels]
        else:
            # color_map_main = vis_params.get("patches",{}).get("color_mapping",{}) # Use patch color map for main particle
            # Using the direct particle color mapping as derived in init
            main_color_mapping = patches_vis_p.get("color_mapping",{}) # int keys
            current_colors = [main_color_mapping.get(t, 'grey') for t in types_f]


        particle_scatter.set_facecolor(current_colors)


        for i in range(max_hist_particles):
            if i < num_p_f and orient_line_p.get("enabled", True):
                l = orient_line_p.get("length",1.0)
                end_pt = pos_f[i] + l * np.array([np.cos(orient_f[i]), np.sin(orient_f[i])])
                orientation_lines_plt[i].set_data([pos_f[i,0], end_pt[0]], [pos_f[i,1], end_pt[1]])
            else: orientation_lines_plt[i].set_data([],[])

        # Patches
        patch_coords, patch_colors_list = [], []
        if patches_vis_p.get("enabled", True) and frame < len(all_patch_data_history) and patches_f_all:
             patch_color_map = patches_vis_p.get("color_mapping",{}) # int keys
             for p_idx_particle in range(num_p_f): # Iterate up to current particles
                 if p_idx_particle < len(patches_f_all): # Check if patch data exists for this particle
                     for patch_data in patches_f_all[p_idx_particle]: # patches_f_all is list of lists
                         patch_coords.append(patch_data["position"])
                         patch_colors_list.append(patch_color_map.get(patch_data["patch_type"], 'black'))
        patch_scatter_plt.set_offsets(np.array(patch_coords) if patch_coords else np.array([]).reshape(0,2) )
        if patch_colors_list : patch_scatter_plt.set_facecolor(patch_colors_list)
        else: patch_scatter_plt.set_facecolor(np.array([]))



        for bline in bond_lines_plt: bline.remove()
        bond_lines_plt.clear()
        if bonds_vis_p.get("enabled", True) and bonds_f and frame < len(all_patch_data_history) and patches_f_all:
            for bond_key in bonds_f.keys():
                is_patch_bond = isinstance(bond_key, tuple) and len(bond_key)==2 and isinstance(bond_key[0],tuple)
                if is_patch_bond:
                    (i,pi), (j,pj) = bond_key
                    if i<num_p_f and j<num_p_f and i<len(patches_f_all) and j<len(patches_f_all) and pi<len(patches_f_all[i]) and pj<len(patches_f_all[j]):
                        pos1, pos2 = patches_f_all[i][pi]["position"], patches_f_all[j][pj]["position"]
                        line, = ax_anim.plot([pos1[0],pos2[0]], [pos1[1],pos2[1]],
                                             color=bonds_vis_p.get("color",'gray'),
                                             lw=bonds_vis_p.get("linewidth", 2.0),
                                             linestyle=bonds_vis_p.get("linestyle", '-'),
                                             zorder=0)
                        bond_lines_plt.append(line)
                # else: # Center-center bonds
                #     i,j = bond_key
                #     if i<num_p_f and j<num_p_f:
                #         line, = ax_anim.plot([pos_f[i,0],pos_f[j,0]], [pos_f[i,1],pos_f[j,1]], color=bonds_vis_p.get("color",'gray'), lw=bonds_vis_p.get("linewidth",1.5), zorder=0)
                #         bond_lines_plt.append(line)


        time_text_plt.set_text(f"Step: {frame}, Time: {frame*particle_parameters['simulation']['dt']:.1f}")
        return [particle_scatter, patch_scatter_plt, time_text_plt] + orientation_lines_plt + bond_lines_plt


    if min_hist_len_anim > 0 :
        ani = animation.FuncAnimation(fig_anim, update_anim, frames=min_hist_len_anim, blit=True, interval=1000/particle_parameters["simulation"]["animation_fps"], repeat=False)
        try:
            print(f"Saving animation to {anim_file}...")
            ani.save(anim_file, writer=animation.PillowWriter(fps=particle_parameters["simulation"]["animation_fps"]))
            print("Animation saved.")
        except Exception as e: print(f"Error saving animation: {e}. Ensure Pillow is installed.")
    plt.close(fig_anim)


# --- Enums, Constants, and Classes from MAUS Prototype ---
# (Adaptive Matrix Solver 0.1)

# --- Enumerations for Problem Types ---
class ProblemType(Enum):
    EIGENVALUE = 1
    SOLVE_LINEAR_SYSTEM = 2
    SVD = 3

# --- Global Configuration Parameters (Informing MAUS's Heuristics) ---
# These are adjustable hyperparameters that MAUS uses in its internal decision-making.
GLOBAL_DEFAULT_PSI_EPSILON_BASE = np.complex128(1e-20) # Base regularization magnitude (multiplied by aggression factor)
GLOBAL_DEFAULT_ALPHA_V_INITIAL = np.complex128(0.01) # Initial learning rate for candidates' steps
GLOBAL_MAX_PSI_ATTEMPTS = 25 # Max attempts for InverseIterateSolver per candidate update
GLOBAL_MAX_STUCK_FOR_RETIREMENT = 8 # Times a candidate can repeatedly fail before being retired (population management)
GLOBAL_MAX_STUCK_FOR_PRUNING = 5 # Used by the population manager, indicates `Fragile` state when avg stuckness is higher than this value
GLOBAL_MIN_WEIGHT_TO_SURVIVE_PRUNE = 1e-10 # Minimum confidence weight for a candidate to stay in population
GLOBAL_VECTOR_SIMILARITY_TOL = 0.999 # Cosine similarity threshold for considering two vectors "the same"
GLOBAL_LAMBDA_SIMILARITY_TOL = 1e-5 # Absolute difference for eigenvalue uniqueness
GLOBAL_SIGMA_SIMILARITY_TOL_ABS = 1e-6 # Absolute threshold for singular value uniqueness (below this, it's considered ~0)
GLOBAL_SIGMA_SIMILARITY_TOL_REL = 1e-4 # Relative threshold for singular value uniqueness (e.g., 1e-4 of max sigma)
GLOBAL_CONVERGENCE_RESIDUAL_TOL = 1e-8 # Default global residual tolerance for MAUS solve.


# --- InverseIterateSolver: Adaptive Local Solver for Ax=b-like Problems ---
# This class encapsulates the robust linear system solving using direct_solve or iterative_gmres,
# dynamically choosing or falling back, and applying Ψ regularization.
class InverseIterateSolver:
    def __init__(self, N, base_psi_epsilon, max_attempts, preferred_method='direct_solve', is_sparse=False):
        self.N = N # Dimension of the matrix for solve
        self.base_psi_epsilon = base_psi_epsilon # Base magnitude for Ψ
        self.max_attempts = max_attempts # Max internal retries for a single solve call
        self.preferred_method = preferred_method # 'direct_solve' (sla.solve or spsolve) or 'iterative_gmres'
        self.fallback_method = 'iterative_gmres' if preferred_method == 'direct_solve' else 'direct_solve' # Auto-determines fallback
        self.is_sparse = is_sparse # True if problem is sparse

    def solve(self, A_target, b_rhs, candidate_stuck_counter):
        """
        Attempts to solve A_target @ x = b_rhs robustly with Psi regularization,
        potentially trying fallback solvers.
        """
        num_psi_attempts = 0
        current_method_for_try = self.preferred_method # Start with preferred method

        while num_psi_attempts < self.max_attempts:
            # Scale PSI by base and attempt count for increasing aggression based on history
            psi_scalar_magnitude = self.base_psi_epsilon * (10**(num_psi_attempts / 2.0)) * (10**(candidate_stuck_counter / 3.0))

            # Create regularization term (Psi): dynamically chooses sparse identity or dense random matrix
            if self.is_sparse:
                regularization_term = sp.identity(self.N, dtype=A_target.dtype, format='csc') * psi_scalar_magnitude
                # Note: `A_target` should already be in a sparse format for addition.
            else: # Dense matrix: adds random noise component to Psi
                random_perturb = (np.random.rand(self.N, self.N) - 0.5 + 1j * (np.random.rand(self.N, self.N) - 0.5)) * psi_scalar_magnitude * 0.15
                regularization_term = psi_scalar_magnitude * np.eye(self.N, dtype=A_target.dtype) + random_perturb

            # Add regularization to the target matrix for solving
            H_solve = A_target + regularization_term

            try:
                # Core solving logic based on `current_method_for_try`
                if current_method_for_try == 'direct_solve':
                    if self.is_sparse:
                        result_vec = spla.spsolve(H_solve.tocsc(), b_rhs) # scipy.sparse.linalg.spsolve for sparse direct solve
                    else:
                        result_vec = sla.solve(H_solve, b_rhs, assume_a='general') # np.linalg.solve for dense direct solve

                elif current_method_for_try == 'iterative_gmres':
                    # GMRES (Generalized Minimal Residual): robust for non-symmetric systems, can handle near-singularity by finding least-squares sol.
                    # It accepts both dense NumPy arrays and sparse SciPy matrices.
                    # x0: initial guess for solution. tol: relative tolerance. maxiter: max iterations.
                    x0_init = b_rhs if b_rhs.shape == H_solve.shape[1:] else np.zeros_like(b_rhs) # Use RHS as initial guess or zeros
                    result_vec, info = spla.gmres(H_solve, b_rhs, x0=x0_init, tol=1e-8, maxiter=50)
                    if info != 0: raise np.linalg.LinAlgError(f"GMRES did not converge cleanly (info={info}).")

                else:
                    raise ValueError(f"Unknown solver method: {current_method_for_try}")

                if not np.all(np.isfinite(result_vec)): # Critical check for NaN/Inf in result
                    raise ValueError("Solution vector not finite after solve.")

                return result_vec, num_psi_attempts # Successful solve and number of attempts

            except (np.linalg.LinAlgError, ValueError, TypeError) as e:
                # If solve fails: first time, switch to fallback method. Subsequent times, retry with stronger Psi.
                if current_method_for_try == self.preferred_method and self.preferred_method != self.fallback_method and num_psi_attempts == 0:
                    current_method_for_try = self.fallback_method # Switch once to fallback on first failure
                    num_psi_attempts = 0 # Reset PSI attempts for the newly chosen solver
                    continue

                num_psi_attempts += 1

        # If all attempts exhausted without success
        raise RuntimeError(f"InverseIterateSolver failed all {self.max_attempts} attempts for {self.preferred_method} and {self.fallback_method}.")


# --- Solution Candidate Class (Represents a single hypothesis/solution candidate) ---
# Each candidate is an autonomous agent making local progress based on MAUS's global strategy.
class SolutionCandidate:
    _candidate_id_counter = 0
    # Internal states define candidate behavior
    class State(Enum):
        EXPLORING = 1  # In search phase, might take larger/randomized steps
        REFINING = 2   # Has found a promising region, focusing on tighter convergence
        STUCK = 3      # Repeatedly failed or diverged locally. Needs intervention or retirement.
        CONVERGED = 4  # Has met convergence criteria
        RETIRED = 5    # Has been pruned from population due to redundancy or persistent failure

    def __init__(self, problem_matrix, problem_type, N_diag, initial_lambda=None, initial_v=None, initial_x=None, initial_u=None, initial_sigma=None, initial_weight=0.01):
        self.id = SolutionCandidate._candidate_id_counter
        SolutionCandidate._candidate_id_counter += 1

        self.N_diag = N_diag # Dimension for square operations (e.g., N for Eigen)
        self.M_rows, self.M_cols = problem_matrix.shape # Actual dimensions of input matrix
        self.problem_type = problem_type
        self.problem_matrix = problem_matrix
        self.b_vector = None

        # Solution parameters (type-specific containers)
        self.lambda_k = initial_lambda
        self.v_k = initial_v
        self.x_k = initial_x
        self.sigma_k = initial_sigma
        self.u_k = initial_u
        self.right_v_k = initial_v

        # Candidate State and confidence tracking
        self.state = SolutionCandidate.State.EXPLORING # Initial state
        self.w_k = initial_weight # Confidence/weight
        self.residual_k = float('inf') # Current residual (lower is better)
        self.prev_residual = float('inf') # Residual from previous step (for adaptation)
        self.alpha_local_step = GLOBAL_DEFAULT_ALPHA_V_INITIAL # Individual step size for reactive gradient

        self.stuck_counter = 0 # Counts how many consecutive times the candidate needed brute-force intervention
        self.local_psi_retries_needed = 0 # Records retries needed by InverseIterateSolver for last update
        self.num_resets = 0 # Counts total times its internal state was randomly re-initialized due to failures

        # History (for debugging and learning over time)
        self.param_history = []
        self.residual_history = []

        self.initialize_random_solution() # Set initial state of solution parameters


    def initialize_random_solution(self):
        # Helper to create a random normalized complex vector
        rand_vec_init = lambda N: (np.random.rand(N) + 1j * np.random.rand(N)).astype(np.complex128)
        # Use simple perturbation or full random based on a threshold on stuck_counter if applicable.
        # For MAUS's internal logic, this just means "reinitialize from scratch if I fail this".
        norm_rand_vec = lambda v_raw: v_raw / np.linalg.norm(v_raw) if np.linalg.norm(v_raw) > 1e-10 else rand_vec_init(v_raw.shape[0]) # Defensive normalization


        if self.problem_type == ProblemType.EIGENVALUE:
            self.v_k = norm_rand_vec(rand_vec_init(self.N_diag))
            self.lambda_k = (random.random() * 5 - 2.5 + 1j * (random.random() * 5 - 2.5)) # Random complex lambda

        elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM:
            self.x_k = norm_rand_vec(rand_vec_init(self.N_diag)) * random.uniform(0.1, 10.0) # Random magnitude initial solution

        elif self.problem_type == ProblemType.SVD:
            self.u_k = norm_rand_vec(rand_vec_init(self.M_rows))
            self.right_v_k = norm_rand_vec(rand_vec_init(self.M_cols))
            self.sigma_k = 1.0

        # Store initial (possibly inf) residual and solution for history.
        self.param_history.append(self.get_current_solution_params())
        self.residual_history.append(self.residual_k)

    def update_solution_step(self, current_matrix_A, b_vector=None, strat_params=None, global_knowledge=None):
        self.b_vector = b_vector # `b` vector for Ax=b problems
        self.prev_residual = self.residual_k # Store last residual for step-size adaptation

        # Parameters for local solver instance from global strategy & knowledge
        overall_psi_aggression_factor = strat_params.get('overall_psi_aggression_factor', 1.0)
        max_psi_retries_global = strat_params.get('max_psi_retries', GLOBAL_MAX_PSI_ATTEMPTS)
        local_solver_preference = global_knowledge.get('local_solver_preference', 'direct_solve') # 'direct_solve' or 'iterative_gmres'
        is_matrix_sparse = global_knowledge.get('is_sparse_problem', False)

        solver_instance = InverseIterateSolver(self.N_diag, GLOBAL_DEFAULT_PSI_EPSILON_BASE * overall_psi_aggression_factor,
                                                max_psi_retries_global, local_solver_preference, is_matrix_sparse)

        # Branch based on problem type for specific update logic
        if self.problem_type == ProblemType.SVD:
            try:
                # SVD works via alternating matrix-vector products (like power method variants).
                # If a vector's norm is tiny, we might add noise or reinitialize.
                if np.linalg.norm(self.right_v_k) < 1e-10:
                    self.right_v_k = (np.random.rand(self.M_cols) + 1j * np.random.rand(self.M_cols)); self.right_v_k /= np.linalg.norm(self.right_v_k)
                    self.stuck_counter += 1; self.num_resets += 1;
                    raise ValueError("SVD right_v_k collapsed. Reinitializing.")

                temp_u_k = current_matrix_A @ self.right_v_k
                self.sigma_k = np.linalg.norm(temp_u_k) # Best singular value estimate
                self.u_k = temp_u_k / (self.sigma_k if self.sigma_k > 1e-10 else 1.0) # Normalize `u`

                if np.linalg.norm(self.u_k) < 1e-10: # Check if u also collapsed (potential error propagation)
                     self.u_k = (np.random.rand(self.M_rows)+1j*np.random.rand(self.M_rows)); self.u_k /= np.linalg.norm(self.u_k)
                     self.stuck_counter += 1; self.num_resets += 1;
                     raise ValueError("SVD u_k collapsed. Reinitializing.")

                temp_v_k = current_matrix_A.conj().T @ self.u_k
                self.sigma_k = max(self.sigma_k, np.linalg.norm(temp_v_k)) # Take the maximum sigma from both updates
                self.right_v_k = temp_v_k / (np.linalg.norm(temp_v_k) if np.linalg.norm(temp_v_k) > 1e-10 else 1.0)

                # Small sigma might indicate convergence to zero singular value, not necessarily a failure.
                if self.sigma_k < GLOBAL_SIGMA_SIMILARITY_TOL_ABS / 100 : # If sigma is tiny even lower than default (often implies it's really 0)
                    self.residual_k = strat_params.get('current_convergence_threshold', 1e-6) * 0.1 # Set very small residual to acknowledge "convergence to zero sigma"
                    self.state = SolutionCandidate.State.CONVERGED # It found a very small sigma and solved for it
                    self.stuck_counter = 0 # No longer stuck
                    # Ensure u and v are well-defined for downstream usage if sigma is zero
                    if np.linalg.norm(self.u_k) < 1e-10: self.u_k = np.ones(self.M_rows, dtype=np.complex128)/np.sqrt(self.M_rows)
                    if np.linalg.norm(self.right_v_k) < 1e-10: self.right_v_k = np.ones(self.M_cols, dtype=np.complex128)/np.sqrt(self.M_cols)

                else: # Otherwise, standard processing
                    self.stuck_counter = max(0, self.stuck_counter - 1) # Reduce stuck counter on successful SVD step

            except (RuntimeError, ValueError, np.linalg.LinAlgError) as e: # Catch any errors from SVD path or its internal vector normalization
                self.stuck_counter += 1; self.w_k *= 0.001; self.alpha_local_step *= 0.5
                self.state = SolutionCandidate.State.STUCK
                if self.stuck_counter >= GLOBAL_MAX_STUCK_FOR_RETIREMENT: self.state = SolutionCandidate.State.RETIRED
                # If SVD method explicitly threw error, re-randomize for brute-force exploration
                self.u_k = (np.random.rand(self.M_rows)+1j*np.random.rand(self.M_rows))/np.sqrt(self.M_rows)
                self.right_v_k = (np.random.rand(self.M_cols)+1j*np.random.rand(self.M_cols))/np.sqrt(self.M_cols)
                self.sigma_k = 1.0

        # --- Common update block for Eigenvalue and SolveLinearSystem problems (using InverseIterateSolver) ---
        else:
            target_A_for_solve = current_matrix_A
            rhs_for_solve = None
            current_main_vec_ref = None

            if self.problem_type == ProblemType.EIGENVALUE:
                if np.linalg.norm(self.v_k) < 1e-10:
                    self.v_k = (np.random.rand(self.N_diag) + 1j*np.random.rand(self.N_diag)); self.v_k /= np.linalg.norm(self.v_k)
                    self.stuck_counter += 1; self.num_resets += 1;
                    raise ValueError("Eigenvector collapsed. Reinitializing random vector for a fresh start.") # This restarts `try` block, potentially with a new `Psi`
                self.lambda_k = (self.v_k.conj().T @ current_matrix_A @ self.v_k) / (self.v_k.conj().T @ self.v_k) # Reactive lambda update
                target_A_for_solve = current_matrix_A - self.lambda_k * np.eye(self.N_diag, dtype=current_matrix_A.dtype)
                rhs_for_solve = self.v_k # `v_k` serves as the right-hand side for (A-λI)z = v
                current_main_vec_ref = self.v_k

            elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM:
                target_A_for_solve = current_matrix_A
                rhs_for_solve = self.b_vector
                current_main_vec_ref = self.x_k

            try:
                new_vec_raw, self.local_psi_retries_needed = solver_instance.solve(target_A_for_solve, rhs_for_solve, self.stuck_counter)

                # Apply alpha_local_step for controlled blend/step. This prevents overshooting and aids stability.
                if self.problem_type == ProblemType.EIGENVALUE:
                    # Blends the old `v_k` with the `new_vec_raw` in the direction of the solution
                    self.v_k = (1.0 - self.alpha_local_step) * self.v_k + self.alpha_local_step * new_vec_raw
                    self.v_k /= np.linalg.norm(self.v_k) if np.linalg.norm(self.v_k) > 1e-10 else (np.random.rand(self.N_diag)+1j*np.random.rand(self.N_diag))/np.sqrt(self.N_diag) # Normalize and protect against 0-norm
                elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM:
                    # For Ax=b, `new_vec_raw` IS the candidate `X`. Alpha blends current `x_k` with this newly calculated `X`.
                    self.x_k = (1.0 - self.alpha_local_step) * current_main_vec_ref + self.alpha_local_step * new_vec_raw

                self.stuck_counter = max(0, self.stuck_counter - 1) # Success means reduction in stuckness

            except (RuntimeError, ValueError) as e: # Catch InverseIterateSolver failure (ran out of PSI/solver types)
                self.stuck_counter += 1
                self.w_k *= 0.001 # Penalize candidate weight
                self.alpha_local_step = max(self.alpha_local_step * 0.5, 1e-6) # Aggressive step size reduction
                if self.stuck_counter >= GLOBAL_MAX_STUCK_FOR_RETIREMENT: # Candidate failed too many times, retire it
                    self.state = SolutionCandidate.State.RETIRED
                    self.num_resets += 1 # Count how many were completely reset and retired
                else: # Otherwise, mark as stuck for now and retry with random state next
                    self.state = SolutionCandidate.State.STUCK
                    self.initialize_random_solution() # Reset vector/params, retaining `stuck_counter`


        # --- Common Residual Calculation & History Logging (Regardless of previous branch) ---
        A = self.problem_matrix # Get the current (potentially updated, e.g., dynamic A(t)) matrix for residual calc
        if self.problem_type == ProblemType.EIGENVALUE:
            self.residual_k = np.linalg.norm(A @ self.v_k - self.lambda_k * self.v_k)
        elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM:
            self.residual_k = np.linalg.norm(A @ self.x_k - self.b_vector)
        # SVD residual is calculated in its update path; just verify it's not infinite now
        elif self.problem_type == ProblemType.SVD:
            self.residual_k = np.linalg.norm(A @ self.right_v_k - self.sigma_k * self.u_k) + \
                              np.linalg.norm(A.conj().T @ self.u_k - self.sigma_k * self.right_v_k)

        # Append to history, now guaranteed to have all parameters from whichever branch
        self.param_history.append(self.get_current_solution_params())
        self.residual_history.append(self.residual_k)

        # Adaptive alpha_local_step and candidate State Transition (Common)
        if self.prev_residual > 1e-10:
            if self.residual_k < self.prev_residual * 0.9: # Significant improvement (reward)
                self.alpha_local_step = min(self.alpha_local_step * 1.1, 1.0)
                if self.state != SolutionCandidate.State.CONVERGED: self.state = SolutionCandidate.State.REFINING
            elif self.residual_k > self.prev_residual * 1.5 and self.prev_residual > GLOBAL_CONVERGENCE_RESIDUAL_TOL * 10: # Diverging significantly, and not already very close to converged
                self.alpha_local_step = max(self.alpha_local_step * 0.5, 1e-6) # Aggressively dampen step size
                if self.state != SolutionCandidate.State.CONVERGED: self.state = SolutionCandidate.State.STUCK # Mark as stuck for current round, allow strategies to handle
            else: # Stagnant or minor progress (decay learning rate, and if it wasn't already in another state)
                self.alpha_local_step = max(self.alpha_local_step * 0.95, 1e-6) # Gradually decrease exploration size
                if self.state not in [SolutionCandidate.State.CONVERGED, SolutionCandidate.State.STUCK, SolutionCandidate.State.RETIRED]:
                     self.state = SolutionCandidate.State.EXPLORING # Continue searching for better paths

        # Final Convergence check (can switch state to CONVERGED)
        if self.residual_k < strat_params.get('current_convergence_threshold', GLOBAL_CONVERGENCE_RESIDUAL_TOL) and \
           np.all(np.isfinite(self.get_current_solution_params()[-1])): # Final check for numerical stability of result
            self.state = SolutionCandidate.State.CONVERGED
            self.w_k = 1.0 # Max confidence for converged solutions
            self.stuck_counter = 0 # Reset stuck counter
            self.alpha_local_step = 0.0 # Halt stepping for converged solutions

    def get_current_solution_params(self):
        # Returns the relevant solution parameters as a tuple
        if self.problem_type == ProblemType.EIGENVALUE: return (self.lambda_k, self.v_k)
        elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM: return (self.x_k,)
        elif self.problem_type == ProblemType.SVD: return (self.sigma_k, self.u_k, self.right_v_k)
        return None

# --- MAUS: The Universal Adaptive Matrix Solver (Main Class) ---
class MAUS_Solver:
    def __init__(self, problem_matrix, problem_type, b_vector=None, initial_num_candidates=None, global_convergence_tol=GLOBAL_CONVERGENCE_RESIDUAL_TOL):
        # Initialize matrix: Convert to sparse if needed, else to dense complex.
        if isinstance(problem_matrix, (sp.spmatrix,)):
            self.M = problem_matrix.copy()
        else:
            self.M = problem_matrix.astype(np.complex128)

        self.N_rows, self.N_cols = self.M.shape
        self.N_diag = self.N_rows # General diagonal dimension placeholder

        self.problem_type = problem_type
        self.b = b_vector.astype(np.complex128) if b_vector is not None else None

        # Initial problem diagnosis to set up `problem_knowledge`
        self.diag_info = self._diagnose_matrix_initial(self.M)
        self.is_sparse_problem_init = self.diag_info['is_sparse_init'] # Is initial matrix sparse (from initial threshold)?
        self.cond_number = self.diag_info['condition_number'] # Initial condition number for dense matrix

        # MAUS's internal "Cognitive State" for the problem: informs all strategy decisions
        self.problem_knowledge = {
            'matrix_type': 'Dense', # Becomes 'Sparse' if converted.
            'spectrum_hint': 'Unknown',
            'numerical_stability_state': 'Stable', # 'Stable', 'Fragile', 'Critical'
            'local_solver_preference': 'direct_solve', # Local solver mode: 'direct_solve' or 'iterative_gmres'
            'effective_rank_SVD': min(self.N_rows, self.N_cols), # SVD rank, estimated dynamically
            'true_matrix_is_singular': self.diag_info['is_singular'], # If initial dense matrix is truly singular
            'is_sparse_problem': self.is_sparse_problem_init # Track actual internal state for sparse solving
        }

        # Convert M to a usable sparse format (CSC for solves) if deemed sparse enough OR is already sparse obj.
        if self.problem_knowledge['is_sparse_problem'] and not isinstance(self.M, (sp.csc_matrix, sp.csr_matrix, sp.coo_matrix)):
             self.M = sp.csc_matrix(self.M)
             # print(f"  MAUS converting input matrix to sparse CSC format for efficient compute.")
             self.problem_knowledge['matrix_type'] = 'Sparse' # Update cognitive state

        # Adaptive strategy parameters: These are dynamically tuned by MAUS
        self.strat_params = {
            'overall_psi_aggression_factor': 1.0,
            'max_psi_retries': GLOBAL_MAX_PSI_ATTEMPTS,
            'min_survival_weight': GLOBAL_MIN_WEIGHT_TO_SURVIVE_PRUNE,
            'spawn_rate_multiplier': 1.0,
            'convergence_tolerance': global_convergence_tol,
            'current_convergence_threshold': 1.0, # This threshold changes based on overall progress
        }

        self._set_initial_strategy() # Sets up initial strategy based on matrix diagnosis

        # Candidate population initialization
        initial_num_candidates = initial_num_candidates if initial_num_candidates is not None else (self.N_diag * 3)
        if self.problem_type == ProblemType.SVD:
            initial_num_candidates = max(initial_num_candidates, min(self.N_rows, self.N_cols) * 3)

        self.candidates = []
        for _ in range(initial_num_candidates):
            self.candidates.append(SolutionCandidate(self.M, self.problem_type, self.N_diag)) # Pass sparse matrix

        SolutionCandidate._candidate_id_counter = initial_num_candidates
        # print(f"MAUS Initialized with {initial_num_candidates} candidates for {self.problem_type.name} (Dims={self.N_rows}x{self.N_cols}).")
        # print(f"Initial matrix diagnostics: Cond={self.cond_number:.2e}, MatrixType={self.problem_knowledge['matrix_type']}. Initial MAUS Knowledge: {self.problem_knowledge['numerical_stability_state']}.")

        # Global metrics for MAUS's internal awareness of overall problem state
        self.landscape_energy = 1.0 # Global objective: minimize this
        self.avg_residual = 1.0
        self.avg_stuckness = 0.0
        self.num_distinct_converged_solutions = 0
        self.converged_solutions = [] # Stores final unique converged solutions found


    def _diagnose_matrix_initial(self, matrix):
        """Initial static diagnosis of the matrix at MAUS initialization."""
        is_sparse_init = False
        if isinstance(matrix, (np.ndarray,)): # Check if a dense NumPy array is sparse enough for conversion
            is_sparse_init = np.count_nonzero(matrix) < 0.25 * matrix.size
        elif isinstance(matrix, (sp.spmatrix,)): # Check if it's already a SciPy sparse matrix object
            is_sparse_init = True

        cond_num = np.inf
        matrix_is_singular = False
        try:
            # Condition number check, but only if matrix is not initially flagged as sparse (costly for large N)
            if not is_sparse_init:
                cond_num = np.linalg.cond(matrix)
                if np.isinf(cond_num) or cond_num > 1e15: matrix_is_singular = True
            else: # For sparse matrix, assume condition number from behavior, or specialized norms.
                pass
        except np.linalg.LinAlgError: # Catches errors during condition calculation itself
            cond_num = np.inf
            matrix_is_singular = True

        return {'is_sparse_init': is_sparse_init, 'condition_number': cond_num, 'is_singular': matrix_is_singular}

    def _set_initial_strategy(self):
        """Sets MAUS's initial global strategy based on initial matrix diagnosis."""

        # Determine initial `numerical_stability_state` and solver preference based on matrix properties.
        # This decision flow dictates how aggressive MAUS starts its "brute-force" exploration.
        if self.cond_number > 1e12:
            self.problem_knowledge['numerical_stability_state'] = 'Critical'
            self.strat_params['overall_psi_aggression_factor'] = 50.0 # High aggression from the start
            self.strat_params['max_psi_retries'] = GLOBAL_MAX_PSI_ATTEMPTS * 2 # Double local retry attempts
            self.strat_params['current_convergence_threshold'] = 1e-2 # Loose initial global convergence target
            self.problem_knowledge['local_solver_preference'] = 'iterative_gmres' # Critical problems start with GMRES
        elif self.cond_number > 1e6: # Moderately ill-conditioned at start (

        # Segment 3: Brain Simulation Classes (and completing MAUS._update_global_diagnostics)

        # Continues from previous Segment 2, from within MAUS_Solver._update_global_diagnostics
        elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM:
            # For Ax=b, there is typically one primary solution, so comparison is straightforward.
            if len(self.converged_solutions) > 0 and np.linalg.norm(current_tuple[0] - self.converged_solutions[0][0]) < self.strat_params['convergence_tolerance'] * 100:
                is_distinct = False
        elif self.problem_type == ProblemType.SVD:
            # SVD: first filter for significance, then for distinctness
            max_current_sigma = max(c.sigma_k.real for c in self.candidates if c.sigma_k.real > 0) if any(c.sigma_k.real > 0 for c in self.candidates) else 1.0
            # Filter: if singular value is tiny compared to max, it's effectively zero, not a "distinct" rank.
            if current_tuple[0].real / max_current_sigma < GLOBAL_SIGMA_SIMILARITY_TOL_REL :
                 is_distinct = False # Treat tiny sigmas as non-distinct by this criteria.

            if is_distinct: # If still distinct after significance check, then compare with already-found solutions
                for s_item in self.converged_solutions:
                    s_sigma, s_u, s_v = s_item[0], s_item[1], s_item[2]
                    effective_abs_tol = GLOBAL_SIGMA_SIMILARITY_TOL_ABS
                    effective_rel_tol = s_sigma * GLOBAL_SIGMA_SIMILARITY_TOL_REL
                    if np.abs(current_tuple[0] - s_sigma) < max(effective_abs_tol, effective_rel_tol) and \
                       np.abs(np.vdot(current_tuple[1], s_u)) > GLOBAL_VECTOR_SIMILARITY_TOL and \
                       np.abs(np.vdot(current_tuple[2], s_v)) > GLOBAL_VECTOR_SIMILARITY_TOL:
                        is_distinct = False; break
            current_sigma_magnitudes.append(current_tuple[0].real) # Collect for overall rank detection heuristic

        if is_distinct:
            self.converged_solutions.append(current_tuple)
            self.num_distinct_converged_solutions += 1

    # Sum metrics ONLY for candidates that are NOT converged AND NOT retired
    for c in self.candidates:
        if c.state not in [SolutionCandidate.State.CONVERGED, SolutionCandidate.State.RETIRED]:
            sum_residuals += c.residual_k
            sum_stuck_counters += c.stuck_counter
            sum_confidence += c.w_k

    # Normalize metrics to contribute to Landscape Energy calculation
    # Ensure denominator is not zero
    num_active_and_non_converged = total_active_candidates - num_converged_all_types
    self.avg_residual = sum_residuals / max(1, num_active_and_non_converged)
    self.avg_stuckness = sum_stuck_counters / max(1, num_active_and_non_converged)
    self.avg_confidence_active = sum_confidence / max(1, num_active_and_non_converged)

    # Dynamic Landscape Energy (MAUS's primary global objective to minimize)
    # Penalizes high residual, high stuckness, and not finding enough solutions
    norm_avg_res = self.avg_residual / (self.strat_params['current_convergence_threshold'] * 10 + DEFAULT_EPSILON) # Normalize by a multiple of current tolerance
    norm_avg_stuck = self.avg_stuckness / (GLOBAL_MAX_STUCK_FOR_RETIREMENT * 2 + DEFAULT_EPSILON) # Penalizes historical failures heavily

    target_sols_N_global = self.N_diag # Default for Eigenvalue problems
    if self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM: target_sols_N_global = 1 # Linear system expects 1 unique solution
    elif self.problem_type == ProblemType.SVD:
        # Dynamic SVD rank detection heuristic (updates problem_knowledge['effective_rank_SVD'])
        if len(current_sigma_magnitudes) > 1: # Only estimate rank if at least 2 sigmas exist
            current_sigma_magnitudes_sorted = sorted([s for s in current_sigma_magnitudes if s > 0], reverse=True) # Filter out noise and sort
            if len(current_sigma_magnitudes_sorted) > 0:
                max_sigma_in_set = current_sigma_magnitudes_sorted[0]
                rank_detected = 0
                for s_val in current_sigma_magnitudes_sorted:
                    # Count as part of rank if above a relative threshold of maximum sigma found
                    if s_val / (max_sigma_in_set + DEFAULT_EPSILON) > GLOBAL_SIGMA_SIMILARITY_TOL_REL:
                        rank_detected += 1
            else: rank_detected = 0 # All sigmas appear to be 0 or too small.

            # MAUS learns the rank dynamically by picking the *minimum* rank detected that is consistent.
            # Never exceed actual dimensions, never drop below 1 unless zero singular values.
            self.problem_knowledge['effective_rank_SVD'] = min(rank_detected, min(self.N_rows, self.N_cols)) if rank_detected > 0 else 0
            if self.problem_knowledge['effective_rank_SVD'] == 0 and min(self.N_rows, self.N_cols) > 0: self.problem_knowledge['effective_rank_SVD'] = 1 # Always at least rank 1 unless problem matrix is truly zero

        target_sols_N_global = self.problem_knowledge['effective_rank_SVD']


    # Calculate remaining landscape energy components
    norm_missing_sols = (target_sols_N_global - self.num_distinct_converged_solutions) / max(1, target_sols_N_global) # If 0 solutions expected/found, no 'missing' penalty
    self.landscape_energy = (norm_avg_res * 0.4) + (norm_avg_stuck * 0.3) + \
                            (norm_missing_sols * 0.3)
    self.landscape_energy = max(0.0, min(1.0, self.landscape_energy)) # Clamp energy between 0 and 1

    # MAUS's internal "Cognitive State" update (inference about stability)
    if self.avg_stuckness > GLOBAL_MAX_STUCK_FOR_RETIREMENT * 0.5: # Many candidates are stuck globally
        self.problem_knowledge['numerical_stability_state'] = 'Critical'
    elif self.avg_stuckness > GLOBAL_MAX_STUCK_FOR_PRUNING * 0.5: # Lower level of stuckness indicates `Fragile` state
         self.problem_knowledge['numerical_stability_state'] = 'Fragile'
    else: # Mostly progressing well
         self.problem_knowledge['numerical_stability_state'] = 'Stable'


# This closes the MAUS_Solver._update_global_diagnostics method


# Segment 3 continues with the definition of UnifiedBrainSimulation (already complete, just noting the conceptual place in segmented code)
class UnifiedBrainSimulation:
    # (Rest of UnifiedBrainSimulation class, as previously defined, is placed here)
    # ... (code omitted for brevity in response but conceptually present here)

    # placeholder method to ensure consistency when re-assembling:
    def plot_results(self):
        print("Brain plotting functionality (placeholder in segment mode).")
        # Actual implementation of plot_results and run_full_simulation is detailed later/earlier,
        # but the conceptual placeholder is relevant for overall structure awareness.
    def run_full_simulation(self):
        print("Brain independent simulation functionality (placeholder in segment mode).")

        # Segment 4: Particle Simulation Functions (Core Dynamics)

# Function to Initialize Particle State (remains the same)
def initialize_particles(parameters):
    """
    Initializes the positions, velocities, orientations, angular velocities,
    types, and masses of particles based on specified initial conditions.
    """
    initial_conditions = parameters["initial_conditions"]
    boundaries = parameters["boundaries"]
    x_min, x_max = boundaries["x_min"], boundaries["x_max"]
    y_min, y_max = boundaries["y_min"], boundaries["y_max"]
    box_size = np.array([x_max - x_min, y_max - y_min])

    num_particles_request = initial_conditions["num_particles_request"]
    initial_type = initial_conditions.get("initial_type", 0) # Default type if not specified
    mass_mapping = initial_conditions["mass_mapping"]

    positions = np.zeros((num_particles_request, 2))
    velocities = np.zeros((num_particles_request, 2))
    accelerations = np.zeros((num_particles_request, 2))
    orientations = np.zeros(num_particles_request)
    angular_velocities = np.zeros(num_particles_request)
    angular_accelerations = np.zeros(num_particles_request)
    types = np.full(num_particles_request, initial_type, dtype=int)
    masses = np.array([mass_mapping.get(t, 1.0) for t in types]) # Default mass 1.0 if type not in mapping

    # Get color mapping from the input parameters dictionary
    vis_params_loaded = parameters.get("visualization", {})
    patches_vis_loaded = vis_params_loaded.get("patches", {})
    color_mapping_loaded = patches_vis_loaded.get("color_mapping", {})
    colors = [color_mapping_loaded.get(t, 'gray') for t in types]


    initial_condition_type = initial_conditions["type"]

    if initial_condition_type == "random":
        positions = np.random.rand(num_particles_request, 2) * box_size + np.array([x_min, y_min])
        velocities = np.random.randn(num_particles_request, 2) * initial_conditions.get("initial_velocity_scale", 0.1)
        if initial_conditions.get("initial_orientation_type", "random") == "random":
             orientations = np.random.rand(num_particles_request) * 2 * np.pi
        else:
             orientations = np.full(num_particles_request, initial_conditions.get("initial_orientation_angle", 0.0))
        angular_velocities = np.random.randn(num_particles_request) * initial_conditions.get("initial_angular_velocity_scale", 0.1)


    elif initial_condition_type == "grid":
        grid_size = int(np.ceil(np.sqrt(num_particles_request)))
        if grid_size == 0: grid_size = 1 # Avoid division by zero for num_particles_request = 0
        x_spacing = box_size[0] / grid_size
        y_spacing = box_size[1] / grid_size
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count < num_particles_request:
                    positions[count, 0] = x_min + (i + 0.5) * x_spacing
                    positions[count, 1] = y_min + (j + 0.5) * y_spacing
                    count += 1
        velocities = np.zeros_like(positions)
        if initial_conditions.get("initial_orientation_type", "random") == "random":
             orientations = np.random.rand(num_particles_request) * 2 * np.pi
        else:
             orientations = np.full(num_particles_request, initial_conditions.get("initial_orientation_angle", 0.0))
        angular_velocities = np.zeros_like(orientations)


    elif initial_condition_type == "grid_swirl":
        grid_size = int(np.ceil(np.sqrt(num_particles_request)))
        if grid_size == 0: grid_size = 1
        x_spacing = box_size[0] / grid_size
        y_spacing = box_size[1] / grid_size
        center = np.array([x_min + box_size[0] / 2, y_min + box_size[1] / 2])
        swirl_strength = initial_conditions.get("swirl_strength", 0.5)
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count < num_particles_request:
                    pos = np.array([x_min + (i + 0.5) * x_spacing, y_min + (j + 0.5) * y_spacing])
                    positions[count, :] = pos
                    vec_to_center = center - pos
                    # Create a perpendicular vector for swirling velocity
                    swirl_vel = np.array([-vec_to_center[1], vec_to_center[0]]) * swirl_strength
                    velocities[count, :] = swirl_vel
                    count += 1
        if initial_conditions.get("initial_orientation_type", "random") == "random":
             orientations = np.random.rand(num_particles_request) * 2 * np.pi
        else:
             orientations = np.full(num_particles_request, initial_conditions.get("initial_orientation_angle", 0.0))
        angular_velocities = np.zeros_like(orientations)


    # Assign specific types if a distribution is requested
    if "type_distribution" in initial_conditions:
         type_dist = initial_conditions["type_distribution"]
         # Ensure keys are int if they were strings in JSON
         type_dist_int_keys = {int(k):v for k,v in type_dist.items()}

         if sum(type_dist_int_keys.values()) == num_particles_request:
              current_idx = 0
              for particle_type, count in type_dist_int_keys.items():
                   types[current_idx : current_idx + count] = particle_type
                   current_idx += count
              masses = np.array([mass_mapping.get(t, 1.0) for t in types])
              colors = [color_mapping_loaded.get(t, 'gray') for t in types] # Use the loaded color mapping
         else:
              print("Warning: Sum of particles in type_distribution does not match num_particles_request. Using default initial type.")


    # Initialize bonds (empty initially)
    bonds = {}

    num_particles = num_particles_request

    return positions, velocities, accelerations, bonds, types, masses, colors, orientations, angular_velocities, angular_accelerations, num_particles


# Function to Apply Periodic Boundary Conditions (remains the same)
def apply_periodic_boundary_conditions(positions, box_size, box_min):
    """
    Applies periodic boundary conditions to particle positions.
    """
    return box_min + np.mod(positions - box_min, box_size)


# Function to Update Particle State (Velocity Verlet Integration - adapted)
def update_particle_state(positions, velocities, accelerations, orientations, angular_velocities, angular_accelerations, dt, damping_factor_effective, max_velocity, moments_of_inertia, particle_parameters):
    """
    Updates particle positions, velocities, orientations, and angular velocities
    using the Velocity Verlet integration scheme, applying damping and velocity limits.
    Uses particle_parameters for boundary info.
    """
    boundaries = particle_parameters["boundaries"]
    x_min, x_max = boundaries["x_min"], boundaries["x_max"]
    y_min, y_max = boundaries["y_min"], boundaries["y_max"]
    box_size = np.array([x_max - x_min, y_max - y_min]) # Corrected y_max - y_min
    box_min = np.array([x_min, y_min])

    # Apply damping to current velocities and angular velocities
    current_velocities = velocities * (1.0 - damping_factor_effective)
    current_angular_velocities = angular_velocities * (1.0 - damping_factor_effective) # Applying linear factor for simplicity

    # Step 1 of Velocity Verlet: Update positions and half-step velocities/angular velocities
    positions_half = positions + current_velocities * dt + 0.5 * accelerations * dt**2 # Use accelerations_t
    angular_orientations_half = orientations + current_angular_velocities * dt + 0.5 * angular_accelerations * dt**2

    # Apply periodic boundary conditions to predicted new positions (half step)
    positions_half_pbc = apply_periodic_boundary_conditions(positions_half, box_size, box_min)
    # Keep orientations normalized within [0, 2*pi) at half step
    orientations_half_normalized = np.mod(angular_orientations_half, 2 * np.pi)


    # Apply velocity limiting
    linear_velocity_magnitudes = np.linalg.norm(current_velocities, axis=1)
    exceed_idx = np.where(linear_velocity_magnitudes > max_velocity)[0]
    if len(exceed_idx) > 0:
        current_velocities[exceed_idx, :] = (current_velocities[exceed_idx, :] /
                                      linear_velocity_magnitudes[exceed_idx, np.newaxis]) * max_velocity


    return positions_half_pbc, current_velocities, orientations_half_normalized, current_angular_velocities


# Function to Calculate Forces and Torques (Adapted for Coupled Simulation)
def calculate_forces(positions, velocities, orientations, types, masses, bonds, particle_parameters, current_step, all_patch_data_input, brain_stimulation_per_region, num_conceptual_regions, adaptive_factors_per_particle: np.ndarray):
    """
    Calculates the net linear forces and torques acting on each particle.
    all_patch_data_input is used if valid, otherwise regenerated.
    Returns new all_patch_data.
    """
    # Extract particle parameters (including new potential parameters)
    forces_params = particle_parameters["forces"]
    bonding_params = particle_parameters["bonding"]
    boundaries = particle_parameters["boundaries"]
    density_repulsion_params = particle_parameters["density_repulsion"]
    external_force_params = particle_parameters["external_force"] # Get external force params
    adaptive_params = particle_parameters.get("adaptive_interactions", {})


    C = forces_params["C"]
    cutoff_distance = forces_params["cutoff_distance"] # for center-center pairwise
    short_range_repulsion_strength = forces_params.get("short_range_repulsion_strength", 50.0)


    density_radius = density_repulsion_params["density_radius"]
    density_repulsion_strength = density_repulsion_params["density_repulsion_strength"]
    x_min, x_max = boundaries["x_min"], boundaries["x_max"]
    y_min, y_max = boundaries["y_min"], boundaries["y_max"]
    box_size = np.array([x_max - x_min, y_max - y_min])

    moment_of_inertia_mapping = forces_params["moment_of_inertia_mapping"]

    patch_params = forces_params.get("patch_params", {})
    patch_enabled = patch_params.get("enabled", False)
    patch_definitions = patch_params.get("patch_definitions", {})

    patch_pairwise_potential_params = patch_params.get("patch_pairwise_potential", {})
    patch_pairwise_potential_type = patch_pairwise_potential_params.get("type", "inverse_square_plus_sr")
    inverse_square_strength_matrix = np.array(patch_pairwise_potential_params.get("inverse_square_strength_matrix", [[]]))
    patch_short_range_repulsion_strength = patch_pairwise_potential_params.get("sr_strength", 50.0)
    lj_pairwise_params = patch_pairwise_potential_params.get("lennard_jones", {})
    lj_pairwise_epsilon_matrix = np.array(lj_pairwise_params.get("epsilon_matrix", [[]]))
    lj_pairwise_sigma_matrix = np.array(lj_pairwise_params.get("sigma_matrix", [[]]))
    lj_pairwise_cutoff_factor = lj_pairwise_params.get("cutoff_factor", 2.5)
    sw_pairwise_params = patch_pairwise_potential_params.get("square_well", {})
    sw_pairwise_epsilon_matrix = np.array(sw_pairwise_params.get("epsilon_matrix", [[]]))
    sw_pairwise_sigma_matrix = np.array(sw_pairwise_params.get("sigma_matrix", [[]]))
    sw_pairwise_lambda_matrix = np.array(sw_pairwise_params.get("lambda_matrix", [[]]))
    sw_pairwise_transition_width = sw_pairwise_params.get("transition_width", 0.1)


    patch_bond_potential_params = bonding_params.get("patch_bond_potential", {})
    patch_bond_potential_type = patch_bond_potential_params.get("type", "harmonic")
    patch_bond_distance_harmonic = patch_bond_potential_params.get("harmonic", {}).get("bond_distance", 1.0)
    patch_bond_strength_harmonic = patch_bond_potential_params.get("harmonic", {}).get("bond_strength", 100.0)
    lj_bond_params = patch_bond_potential_params.get("lennard_jones", {})
    lj_bond_epsilon = lj_bond_params.get("epsilon", 5.0)
    lj_bond_sigma = lj_bond_params.get("sigma", 1.5)
    sw_bond_params = patch_bond_potential_params.get("square_well", {})
    sw_bond_epsilon = sw_bond_params.get("epsilon", 5.0)
    sw_bond_sigma = sw_bond_params.get("sigma", 1.5)
    sw_bond_lambda = sw_bond_params.get("lambda", 1.5)
    sw_bond_transition_width = sw_bond_params.get("transition_width", 0.1)
    patch_cutoff_distance_param = patch_params.get("patch_cutoff_distance", 5.0)


    orientation_potential_params = forces_params.get("orientation_potential", {})
    bond_angle_potential_params = orientation_potential_params.get("bond_angle_potential", {})
    bond_angle_potential_enabled = bond_angle_potential_params.get("enabled", False)
    bond_angle_strength = bond_angle_potential_params.get("strength", 0.0)
    ideal_angle_mapping = bond_angle_potential_params.get("ideal_angle_mapping", {})


    adaptive_enabled = adaptive_params.get("enabled", False)
    bond_strength_adaptation_enabled = adaptive_params.get("bond_strength_adaptation", {}).get("enabled", False)
    adaptation_rate = adaptive_params.get("bond_strength_adaptation", {}).get("adaptation_rate", 0.0)
    target_strength = adaptive_params.get("bond_strength_adaptation", {}).get("target_strength", bonding_params.get("bond_strength", 100.0))


    num_particles = positions.shape[0]
    if num_particles == 0: return np.array([]), np.array([]), []
    net_linear_forces = np.zeros_like(positions)
    net_torques = np.zeros(num_particles)

    # Regenerate all_patch_data based on current positions and orientations for this force calculation
    current_all_patch_data = []
    if patch_enabled and patch_definitions:
        for i in range(num_particles):
            particle_type_int = types[i] # types are int
            # Defensive check for patch_definitions key presence
            pdefs_i = [] # Initialize to empty list
            if particle_type_int in patch_definitions:
                 pdefs_i = patch_definitions.get(particle_type_int)
            elif str(particle_type_int) in patch_definitions: # Fallback to string key if int key not found
                 pdefs_i = patch_definitions.get(str(particle_type_int))

            # Ensure it is a list before iteration
            if not isinstance(pdefs_i, list):
                pdefs_i = []


            particle_patches_data = []
            for patch_index_on_particle, patch_spec in enumerate(pdefs_i): # Use pdefs_i here
                p_dist = patch_spec.get("distance", 0.0)
                p_angle_rel = patch_spec.get("angle_relative_to_particle", 0.0)
                patch_type = patch_spec.get("patch_type", 0)
                total_patch_angle = orientations[i] + p_angle_rel
                patch_offset = np.array([p_dist * np.cos(total_patch_angle), p_dist * np.sin(total_patch_angle)])
                patch_position = positions[i, :] + patch_offset
                particle_patches_data.append({
                    "position": patch_position, "patch_type": patch_type,
                    "particle_index": i, "patch_index_on_particle": patch_index_on_particle
                })
            current_all_patch_data.append(particle_patches_data)


    # --- Central Force & Density Repulsion ---
    box_center = np.array([x_min + box_size[0]/2, y_min + box_size[1]/2])
    vec_to_center_of_box = box_center - positions
    central_linear_forces = C * vec_to_center_of_box # Force towards center
    net_linear_forces += central_linear_forces

    # Density Repulsion (away from box center, scaled by local density)
    if density_repulsion_strength > 0 and density_radius > 0 :
        r_center_to_center = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        r_center_to_center_pbc = r_center_to_center - box_size * np.round(r_center_to_center / box_size)
        r_center_mag = np.sqrt(np.sum(r_center_to_center_pbc**2, axis=2))
        np.fill_diagonal(r_center_mag, np.inf) # Ignore self-distance

        local_density_counts = np.sum(r_center_mag < density_radius, axis=1)
        direction_from_center = positions - box_center # Vector from center to particle
        direction_from_center_mag = np.linalg.norm(direction_from_center, axis=1)
        # Avoid division by zero if a particle is exactly at the center
        safe_mag = np.where(direction_from_center_mag < DEFAULT_EPSILON, 1.0, direction_from_center_mag)
        density_force_direction = direction_from_center / safe_mag[:, np.newaxis]
        density_force_magnitude = density_repulsion_strength * local_density_counts
        density_linear_forces = density_force_magnitude[:, np.newaxis] * density_force_direction
        net_linear_forces += density_linear_forces


    # --- Patch-Based Forces (Pairwise Non-Bonded & Bonded) ---
    if patch_enabled and patch_definitions and current_all_patch_data:
        # Determine generous search cutoff considering max patch extension
        max_patch_extension = 0
        for p_type, p_defs in patch_definitions.items():
            for p_spec in p_defs:
                max_patch_extension = max(max_patch_extension, p_spec.get("distance",0))
        patch_search_cutoff_dynamic = patch_cutoff_distance_param + 2 * max_patch_extension

        for i in range(num_particles):
            for j in range(i + 1, num_particles):
                r_ij_center_vec = positions[i, :] - positions[j, :]
                r_ij_center_vec_pbc = r_ij_center_vec - box_size * np.round(r_ij_center_vec / box_size)
                r_ij_center_mag = np.linalg.norm(r_ij_center_vec_pbc)

                if r_ij_center_mag > patch_search_cutoff_dynamic: continue
                if not current_all_patch_data[i] or not current_all_patch_data[j]: continue


                for patch_i_data in current_all_patch_data[i]:
                    for patch_j_data in current_all_patch_data[j]:
                        # patch_i_data structure: {"position", "patch_type", "particle_index", "patch_index_on_particle"}
                        p_i_idx_on_particle = patch_i_data["patch_index_on_particle"]
                        p_j_idx_on_particle = patch_j_data["patch_index_on_particle"]
                        bond_candidate_key = tuple(sorted(((i, p_i_idx_on_particle), (j, p_j_idx_on_particle))))

                        patch_i_pos = patch_i_data["position"]
                        patch_j_pos = patch_j_data["position"]
                        patch_i_type = patch_i_data["patch_type"]
                        patch_j_type = patch_j_data["patch_type"]

                        r_patch_raw = patch_i_pos - patch_j_pos
                        r_patch_pbc = r_patch_raw - box_size * np.round(r_patch_raw / box_size)
                        r_patch_mag = np.linalg.norm(r_patch_pbc)
                        if r_patch_mag < DEFAULT_EPSILON: # Overlapping patches
                             # Apply a large repulsion and random nudge
                             rand_nudge = normalize_vector_np(np.random.rand(2)-0.5) * DEFAULT_EPSILON * 10
                             net_linear_forces[i,:] += rand_nudge * short_range_repulsion_strength * 100
                             net_linear_forces[j,:] -= rand_nudge * short_range_repulsion_strength * 100
                             continue

                        patch_to_patch_direction = r_patch_pbc / r_patch_mag
                        force_magnitude_on_patch_i = 0.0


                        if bond_candidate_key in bonds: # --- BONDED PATCH INTERACTION ---
                            current_bond_strength_h = patch_bond_strength_harmonic
                            if adaptive_enabled and bond_strength_adaptation_enabled:
                                bond_info = bonds[bond_candidate_key]
                                formed_step = bond_info.get('formed_step', 0)
                                initial_strength = bond_info.get('initial_strength', patch_bond_strength_harmonic)
                                bond_age = current_step - formed_step
                                current_bond_strength_h = initial_strength + bond_age * adaptation_rate
                                current_bond_strength_h = np.clip(current_bond_strength_h, 0, target_strength if adaptation_rate >0 else np.inf)


                            if patch_bond_potential_type == "harmonic":
                                force_magnitude_on_patch_i = -current_bond_strength_h * (r_patch_mag - patch_bond_distance_harmonic)
                            elif patch_bond_potential_type == "lennard_jones":
                                r6 = (lj_bond_sigma / r_patch_mag)**6; r12 = r6**2
                                force_magnitude_on_patch_i = -24 * lj_bond_epsilon / r_patch_mag * (2 * r12 - r6)
                            elif patch_bond_potential_type == "square_well":
                                force_magnitude_on_patch_i = smoothed_square_well_force(r_patch_mag, sw_bond_sigma, sw_bond_lambda, sw_bond_epsilon, sw_bond_transition_width)
                        else: # --- NON-BONDED PATCH PAIRWISE INTERACTION ---
                            if r_patch_mag < patch_cutoff_distance_param : # General cutoff for pairwise
                                if patch_pairwise_potential_type == "inverse_square_plus_sr":
                                    interaction_k = 0.0
                                    if 0<=patch_i_type<inverse_square_strength_matrix.shape[0] and 0<=patch_j_type<inverse_square_strength_matrix.shape[1]:
                                        interaction_k = inverse_square_strength_matrix[patch_i_type, patch_j_type]
                                    f_inv_sq = interaction_k / r_patch_mag**2
                                    f_sr = patch_short_range_repulsion_strength * (1/r_patch_mag - 1/patch_cutoff_distance_param) if r_patch_mag < patch_cutoff_distance_param else 0
                                    force_magnitude_on_patch_i = -(f_inv_sq + f_sr) # This needs potential (NOT derivative), so change here
                                elif patch_pairwise_potential_type == "lennard_jones":
                                    eps, sig = 0.0, 1.0
                                    if 0<=patch_i_type<lj_pairwise_epsilon_matrix.shape[0] and 0<=patch_j_type<lj_pairwise_epsilon_matrix.shape[1]:
                                        eps = lj_pairwise_epsilon_matrix[patch_i_type, patch_j_type]
                                    if 0<=patch_i_type<lj_pairwise_sigma_matrix.shape[0] and 0<=patch_j_type<lj_pairwise_sigma_matrix.shape[1]:
                                        sig = lj_pairwise_sigma_matrix[patch_i_type, patch_j_type]
                                    cutoff_lj = lj_pairwise_cutoff_factor * sig
                                    if r_patch_mag < cutoff_lj:
                                        r6 = (sig / r_patch_mag)**6; r12 = r6**2
                                        force_magnitude_on_patch_i = -24 * eps / r_patch_mag * (2 * r12 - r6)
                                elif patch_pairwise_potential_type == "square_well":
                                    eps, sig, lam = 0.0, 1.0, 1.5
                                    if 0<=patch_i_type<sw_pairwise_epsilon_matrix.shape[0] and 0<=patch_j_type<sw_pairwise_epsilon_matrix.shape[1]:
                                        eps = sw_pairwise_epsilon_matrix[patch_i_type, patch_j_type]
                                    if 0<=patch_i_type<sw_pairwise_sigma_matrix.shape[0] and 0<=patch_j_type<sw_pairwise_sigma_matrix.shape[1]:
                                        sig = sw_pairwise_sigma_matrix[patch_i_type, patch_j_type]
                                    if 0<=patch_i_type<sw_pairwise_lambda_matrix.shape[0] and 0<=patch_j_type<sw_pairwise_lambda_matrix.shape[1]:
                                        lam = sw_pairwise_lambda_matrix[patch_i_type, patch_j_type]
                                    force_magnitude_on_patch_i = smoothed_square_well_force(r_patch_mag, sig, lam, eps, sw_pairwise_transition_width)


                        # Apply force and torque from patch interaction
                        force_on_patch_i_vec = force_magnitude_on_patch_i * patch_to_patch_direction
                        net_linear_forces[i, :] += force_on_patch_i_vec
                        net_linear_forces[j, :] -= force_on_patch_i_vec

                        r_vec_i_to_patch_i = patch_i_pos - positions[i, :]
                        torque_on_i = r_vec_i_to_patch_i[0] * force_on_patch_i_vec[1] - r_vec_i_to_patch_i[1] * force_on_patch_i_vec[0]
                        net_torques[i] += torque_on_i

                        r_vec_j_to_patch_j = patch_j_pos - positions[j, :]
                        torque_on_j = r_vec_j_to_patch_j[0] * (-force_on_patch_i_vec[1]) - r_vec_j_to_patch_j[1] * (-force_on_patch_i_vec[0])
                        net_torques[j] += torque_on_j


    # --- External Forces from Brain Stimulation ---
    if external_force_params.get("enabled", False) and brain_stimulation_per_region is not None and num_conceptual_regions > 0:
         brain_to_particle_force_scale = default_params_unified['brain_stimulation_to_particle_force_scale']
         region_width_particle_space = box_size[0] / num_conceptual_regions # Assume regions along x-axis

         for i in range(num_particles):
              particle_x_relative = positions[i, 0] - x_min
              region_idx = int(particle_x_relative / region_width_particle_space)
              region_idx = np.clip(region_idx, 0, num_conceptual_regions - 1)

              stimulation_level = brain_stimulation_per_region[region_idx]
              # Corrected logic from identified point 1
              region_center_x_abs = x_min + (region_idx + 0.5) * region_width_particle_space
              force_dir_to_region_center = np.array([region_center_x_abs - positions[i,0], 0.0]) # Force only in X
              force_dir_mag = np.linalg.norm(force_dir_to_region_center)
              force_dir_norm = force_dir_to_region_center / force_dir_mag if force_dir_mag > DEFAULT_EPSILON else np.array([0.0,0.0])

              force_magnitude_from_brain = stimulation_level * brain_to_particle_force_scale
              external_force_on_particle = force_magnitude_from_brain * force_dir_norm
              net_linear_forces[i, :] += external_force_on_particle


    # --- Friction (Applied per-particle, scaled by adaptive_factors_per_particle) ---
    if external_force_params.get("friction_enabled", False) and adaptive_factors_per_particle is not None and adaptive_factors_per_particle.shape[0] == num_particles:
        friction_coeff = external_force_params.get("friction_coefficient", 0.1)
        # Apply zone-specific adaptive factor to individual particles.
        per_particle_friction = friction_coeff * adaptive_factors_per_particle # assuming 1-to-1 particle-to-factor mapping is possible here
        net_linear_forces -= per_particle_friction[:, np.newaxis] * masses[:, np.newaxis] * velocities # mass-proportional friction
    if external_force_params.get("angular_friction_enabled", False) and adaptive_factors_per_particle is not None and adaptive_factors_per_particle.shape[0] == num_particles:
        angular_friction_coeff = external_force_params.get("angular_friction_coefficient", 0.05)
        moments_of_inertia_arr = np.array([moment_of_inertia_mapping.get(t, 1.0) for t in types])
        per_particle_angular_friction = angular_friction_coeff * adaptive_factors_per_particle # apply factor for each particle based on its zone
        net_torques -= per_particle_angular_friction * moments_of_inertia_arr * angular_velocities # MoI-proportional

    return net_linear_forces, net_torques, current_all_patch_data


# Function to Calculate Total Energy (Adapted for Coupled Simulation)
def calculate_total_energy(positions, velocities, angular_velocities, masses, bonds, types, orientations, particle_parameters, current_step, all_patch_data):
    """
    Calculates the total energy of the system, including kinetic, pairwise potential,
    bond potential, central potential, and orientation potential energy.
    Uses selectable potential types (including Smoothed Square Well).
    Requires current step and all_patch_data for adaptive interactions and patch positions.

    Args:
        positions (np.ndarray): Current positions of particles.
        velocities (np.ndarray): Current linear velocities.
        angular_velocities (np.ndarray): Current angular velocities.
        masses (np.ndarray): Masses of particles.
        bonds (dict): Current dictionary of active bonds.
        types (np.ndarray): Types of particles.
        orientations (np.ndarray): Current orientations of particles.
        particle_parameters (dict): Dictionary of particle simulation parameters.
        current_step (int): The current simulation step number.
        all_patch_data (list of lists): Data structure containing info for all patches.


    Returns:
        float: The total energy of the system.
    """
    # Extract particle parameters (including new potential parameters)
    forces_params = particle_parameters["forces"]
    bonding_params = particle_parameters["bonding"]
    boundaries = particle_parameters["boundaries"]
    # density_repulsion_params = particle_parameters["density_repulsion"] # Not typically included in potential energy
    adaptive_params = particle_parameters.get("adaptive_interactions", {})

    C = forces_params["C"]
    # cutoff_distance = forces_params["cutoff_distance"] # For center-center pairwise (if used)

    # bond_distance_param = bonding_params.get("patch_bond_potential", {}).get("harmonic", {}).get("bond_distance", bonding_params.get("bond_distance", 1.0))
    # bond_strength_param = bonding_params.get("patch_bond_potential", {}).get("harmonic", {}).get("bond_strength", bonding_params.get("bond_strength", 100.0))

    x_min, x_max = boundaries["x_min"], boundaries["x_max"]
    y_min, y_max = boundaries["y_min"], boundaries["y_max"]
    box_size = np.array([x_max - x_min, y_max - y_min])
    box_center = np.array([x_min + box_size[0]/2, y_min + box_size[1]/2])


    moment_of_inertia_mapping = forces_params["moment_of_inertia_mapping"]

    patch_params = forces_params.get("patch_params", {})
    patch_enabled = patch_params.get("enabled", False)
    patch_definitions = patch_params.get("patch_definitions", {})

    patch_pairwise_potential_params = patch_params.get("patch_pairwise_potential", {})
    patch_pairwise_potential_type = patch_pairwise_potential_params.get("type", "inverse_square_plus_sr")

    inverse_square_strength_matrix = np.array(patch_pairwise_potential_params.get("inverse_square_strength_matrix", [[]]))
    patch_sr_strength = patch_pairwise_potential_params.get("sr_strength", 50.0) # For inverse_square_plus_sr

    lj_pairwise_params = patch_pairwise_potential_params.get("lennard_jones", {})
    lj_pairwise_epsilon_matrix = np.array(lj_pairwise_params.get("epsilon_matrix", [[]]))
    lj_pairwise_sigma_matrix = np.array(lj_pairwise_params.get("sigma_matrix", [[]]))
    lj_pairwise_cutoff_factor = lj_pairwise_params.get("cutoff_factor", 2.5)

    sw_pairwise_params = patch_pairwise_potential_params.get("square_well", {})
    sw_pairwise_epsilon_matrix = np.array(sw_pairwise_params.get("epsilon_matrix", [[]]))
    sw_pairwise_sigma_matrix = np.array(sw_pairwise_params.get("sigma_matrix", [[]]))
    sw_pairwise_lambda_matrix = np.array(sw_pairwise_params.get("lambda_matrix", [[]]))
    sw_pairwise_transition_width = sw_pairwise_params.get("transition_width", 0.1)


    patch_bond_potential_params = bonding_params.get("patch_bond_potential", {})
    patch_bond_potential_type = patch_bond_potential_params.get("type", "harmonic")

    patch_bond_distance_harmonic = patch_bond_potential_params.get("harmonic", {}).get("bond_distance", 1.0)
    patch_bond_strength_harmonic = patch_bond_potential_params.get("harmonic", {}).get("bond_strength", 100.0)

    lj_bond_params = patch_bond_potential_params.get("lennard_jones", {})
    lj_bond_epsilon = lj_bond_params.get("epsilon", 5.0)
    lj_bond_sigma = lj_bond_params.get("sigma", 1.5)

    sw_bond_params = patch_bond_potential_params.get("square_well", {})
    sw_bond_epsilon = sw_bond_params.get("epsilon", 5.0)
    sw_bond_sigma = sw_bond_params.get("sigma", 1.5)
    sw_bond_lambda = sw_bond_params.get("lambda", 1.5)
    sw_bond_transition_width = sw_bond_params.get("transition_width", 0.1)


    patch_cutoff_distance_param = patch_params.get("patch_cutoff_distance", 5.0)


    orientation_potential_params = forces_params.get("orientation_potential", {})
    bond_angle_potential_params = orientation_potential_params.get("bond_angle_potential", {})
    bond_angle_potential_enabled = bond_angle_potential_params.get("enabled", False)
    bond_angle_strength = bond_angle_potential_params.get("strength", 0.0)
    ideal_angle_mapping = bond_angle_potential_params.get("ideal_angle_mapping", {}) # keys are int

    adaptive_enabled = adaptive_params.get("enabled", False)
    bond_strength_adaptation_enabled = adaptive_params.get("bond_strength_adaptation", {}).get("enabled", False)
    adaptation_rate = adaptive_params.get("bond_strength_adaptation", {}).get("adaptation_rate", 0.0)
    target_strength = adaptive_params.get("bond_strength_adaptation", {}).get("target_strength", bonding_params.get("bond_strength", 100.0))


    num_particles = positions.shape[0]
    if num_particles == 0: return 0.0

    # --- Kinetic Energy ---
    linear_kinetic_energy = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
    moments_of_inertia_array = np.array([moment_of_inertia_mapping.get(t, 1.0) for t in types])
    angular_kinetic_energy = 0.5 * np.sum(moments_of_inertia_array * angular_velocities**2)
    kinetic_energy = linear_kinetic_energy + angular_kinetic_energy

    # --- Potential Energy ---
    potential_energy = 0.0
    # Central Potential: U = 0.5 * C * sum(dist_from_center_i^2)
    dist_from_center_sq = np.sum((positions - box_center)**2, axis=1)
    potential_energy += 0.5 * C * np.sum(dist_from_center_sq)


    # --- Patch-Based Potential Energy (Pairwise Non-Bonded & Bonded) ---
    if patch_enabled and patch_definitions and all_patch_data:
        max_patch_extension = 0
        for p_type, p_defs in patch_definitions.items():
            for p_spec in p_defs: max_patch_extension = max(max_patch_extension, p_spec.get("distance",0))
        patch_search_cutoff_dynamic = patch_cutoff_distance_param + 2 * max_patch_extension


        for i in range(num_particles):
            for j in range(i + 1, num_particles):
                r_ij_center_vec = positions[i, :] - positions[j, :]
                r_ij_center_vec_pbc = r_ij_center_vec - box_size * np.round(r_ij_center_vec / box_size)
                r_ij_center_mag = np.linalg.norm(r_ij_center_vec_pbc)
                if r_ij_center_mag > patch_search_cutoff_dynamic: continue
                if not all_patch_data[i] or not all_patch_data[j]: continue


                for patch_i_data in all_patch_data[i]:
                    for patch_j_data in all_patch_data[j]:
                        p_i_idx_on_particle = patch_i_data["patch_index_on_particle"]
                        p_j_idx_on_particle = patch_j_data["patch_index_on_particle"]
                        bond_candidate_key = tuple(sorted(((i, p_i_idx_on_particle), (j, p_j_idx_on_particle))))

                        patch_i_pos = patch_i_data["position"]
                        patch_j_pos = patch_j_data["position"]
                        patch_i_type = patch_i_data["patch_type"]
                        patch_j_type = patch_j_data["patch_type"]

                        r_patch_raw = patch_i_pos - patch_j_pos
                        r_patch_pbc = r_patch_raw - box_size * np.round(r_patch_raw / box_size)
                        r_patch_mag = np.linalg.norm(r_patch_pbc)
                        if r_patch_mag < DEFAULT_EPSILON: potential_energy += np.inf; continue


                        U_ij_patch = 0.0
                        if bond_candidate_key in bonds: # --- BONDED PATCH POTENTIAL ---
                            current_bond_strength_h = patch_bond_strength_harmonic
                            if adaptive_enabled and bond_strength_adaptation_enabled:
                                bond_info = bonds[bond_candidate_key]
                                formed_step = bond_info.get('formed_step', 0)
                                initial_strength = bond_info.get('initial_strength', patch_bond_strength_harmonic)
                                bond_age = current_step - formed_step
                                current_bond_strength_h = initial_strength + bond_age * adaptation_rate
                                current_bond_strength_h = np.clip(current_bond_strength_h, 0, target_strength if adaptation_rate >0 else np.inf)

                            if patch_bond_potential_type == "harmonic":
                                U_ij_patch = 0.5 * current_bond_strength_h * (r_patch_mag - patch_bond_distance_harmonic)**2
                            elif patch_bond_potential_type == "lennard_jones":
                                r6 = (lj_bond_sigma / r_patch_mag)**6; r12 = r6**2
                                U_ij_patch = 4 * lj_bond_epsilon * (r12 - r6)
                            elif patch_bond_potential_type == "square_well":
                                U_ij_patch = smoothed_square_well_potential(r_patch_mag, sw_bond_sigma, sw_bond_lambda, sw_bond_epsilon, sw_bond_transition_width)
                        else: # --- NON-BONDED PATCH PAIRWISE POTENTIAL ---
                            if r_patch_mag < patch_cutoff_distance_param :
                                if patch_pairwise_potential_type == "inverse_square_plus_sr":
                                    interaction_k = 0.0
                                    if 0<=patch_i_type<inverse_square_strength_matrix.shape[0] and 0<=patch_j_type<inverse_square_strength_matrix.shape[1]:
                                        interaction_k = inverse_square_strength_matrix[patch_i_type, patch_j_type]
                                    # Correction from previous cut-off: Use Potential calculation directly
                                    U_inv_sq = -interaction_k / r_patch_mag
                                    # SR potential: A/r for A.k.a inverse linear (like Coulomb) potential is -k*log(r). But needs proper bounds.
                                    # Previous formula had `patch_short_range_repulsion_strength * (1/r_patch_mag - 1/patch_cutoff_distance_param)` as `f_sr`, a force.
                                    # The original force for `1/r_patch_mag` is `log(r)`.
                                    # This might have been incorrect in original formulation; will rely on forces as derived correctly there
                                    # For energy: `U_sr` will be `sr_strength * (log(r) - r/rcut)` with positive sr_strength for repulsion
                                    U_sr = patch_sr_strength * (math.log(max(r_patch_mag, DEFAULT_EPSILON)) - r_patch_mag / patch_cutoff_distance_param)
                                    U_ij_patch = U_inv_sq + U_sr
                                elif patch_pairwise_potential_type == "lennard_jones":
                                    eps, sig = 0.0,1.0
                                    if 0<=patch_i_type<lj_pairwise_epsilon_matrix.shape[0] and 0<=patch_j_type<lj_pairwise_epsilon_matrix.shape[1]: eps = lj_pairwise_epsilon_matrix[patch_i_type, patch_j_type]
                                    if 0<=patch_i_type<lj_pairwise_sigma_matrix.shape[0] and 0<=patch_j_type<lj_pairwise_sigma_matrix.shape[1]: sig = lj_pairwise_sigma_matrix[patch_i_type, patch_j_type]
                                    cutoff_lj = lj_pairwise_cutoff_factor * sig
                                    if r_patch_mag < cutoff_lj:
                                        r6 = (sig / r_patch_mag)**6; r12 = r6**2
                                        U_ij_patch = 4 * eps * (r12 - r6)
                                        rc6 = (sig / cutoff_lj)**6; rc12 = rc6**2 # Shift potential
                                        U_ij_patch -= 4 * eps * (rc12 - rc6)
                                elif patch_pairwise_potential_type == "square_well":
                                    eps, sig, lam = 0.0,1.0,1.5
                                    if 0<=patch_i_type<sw_pairwise_epsilon_matrix.shape[0] and 0<=patch_j_type<sw_pairwise_epsilon_matrix.shape[1]: eps = sw_pairwise_epsilon_matrix[patch_i_type, patch_j_type]
                                    if 0<=patch_i_type<sw_pairwise_sigma_matrix.shape[0] and 0<=patch_j_type<sw_pairwise_sigma_matrix.shape[1]: sig = sw_pairwise_sigma_matrix[patch_i_type, patch_j_type]
                                    if 0<=patch_i_type<sw_pairwise_lambda_matrix.shape[0] and 0<=patch_j_type<sw_pairwise_lambda_matrix.shape[1]: lam = sw_pairwise_lambda_matrix[patch_i_type, patch_j_type]
                                    U_ij_patch = smoothed_square_well_potential(r_patch_mag, sig, lam, eps, sw_pairwise_transition_width)
                        potential_energy += U_ij_patch

    # --- Orientation Potential Energy ---
    if patch_enabled and bond_angle_potential_enabled and all_patch_data:
        for bond_key in bonds.keys():
            if not (isinstance(bond_key, tuple) and len(bond_key) == 2 and isinstance(bond_key[0], tuple)): continue
            (i, p_idx_i), (j, p_idx_j) = bond_key
            if not (i < num_particles and j < num_particles and all_patch_data[i] and all_patch_data[j] and \
                    p_idx_i < len(all_patch_data[i]) and p_idx_j < len(all_patch_data[j])): continue

            patch_i_data = all_patch_data[i][p_idx_i]
            patch_j_data = all_patch_data[j][p_idx_j]
            patch_i_pos, patch_j_pos = patch_i_data["position"], patch_j_data["position"]
            patch_i_type, patch_j_type = patch_i_data["patch_type"], patch_j_data["patch_type"]

            r_patch_raw = patch_j_pos - patch_i_pos
            r_patch_pbc = r_patch_raw - box_size * np.round(r_patch_raw / box_size)
            r_patch_mag = np.linalg.norm(r_patch_pbc)
            if r_patch_mag < DEFAULT_EPSILON: continue
            bond_vec_dir = r_patch_pbc / r_patch_mag

            particle_type_i_int = types[i] # Get particle type (int)
            # Defensive initialization for pdefs_i
            pdefs_i = []
            if particle_type_i_int in patch_definitions:
                pdefs_i = patch_definitions.get(particle_type_i_int)
            elif str(particle_type_i_int) in patch_definitions:
                pdefs_i = patch_definitions.get(str(particle_type_i_int))
            if not isinstance(pdefs_i, list): pdefs_i = [] # Final safeguard

            if p_idx_i >= len(pdefs_i): continue
            angle_rel_i = pdefs_i[p_idx_i].get("angle_relative_to_particle",0.0)
            abs_angle_i = orientations[i] + angle_rel_i
            patch_i_orient_vec = np.array([np.cos(abs_angle_i), np.sin(abs_angle_i)])
            ideal_angle_i_rad = ideal_angle_mapping.get(patch_i_type, 0.0) # ideal_angle_mapping keys are int

            current_angle_i_to_bond = np.arctan2(patch_i_orient_vec[0]*bond_vec_dir[1] - patch_i_orient_vec[1]*bond_vec_dir[0], np.dot(patch_i_orient_vec, bond_vec_dir))
            angle_dev_i_sq = (np.mod(current_angle_i_to_bond - ideal_angle_i_rad + np.pi, 2 * np.pi) - np.pi)**2


            particle_type_j_int = types[j]
            # Defensive initialization for pdefs_j
            pdefs_j = []
            if particle_type_j_int in patch_definitions:
                pdefs_j = patch_definitions.get(particle_type_j_int)
            elif str(particle_type_j_int) in patch_definitions:
                pdefs_j = patch_definitions.get(str(particle_type_j_int))
            if not isinstance(pdefs_j, list): pdefs_j = [] # Final safeguard

            if p_idx_j >= len(pdefs_j): continue
            angle_rel_j = pdefs_j[p_idx_j].get("angle_relative_to_particle",0.0)
            abs_angle_j = orientations[j] + angle_rel_j
            patch_j_orient_vec = np.array([np.cos(abs_angle_j), np.sin(abs_angle_j)])
            ideal_j_rad = ideal_angle_mapping.get(patch_j_type, 0.0)

            current_angle_j_to_bond = np.arctan2(patch_j_orient_vec[0]*(-bond_vec_dir[1])-patch_j_orient_vec[1]*(-bond_vec_dir[0]), np.dot(patch_j_orient_vec,-bond_vec_dir))
            angle_dev_j_sq = (np.mod(current_angle_j_to_bond - ideal_j_rad + np.pi, 2*np.pi) - np.pi)**2

            potential_energy += 0.5 * bond_angle_strength * (angle_dev_i_sq + angle_dev_j_sq)


    return kinetic_energy + potential_energy


# Function to Update Bonds (remains the same, uses particle_parameters)
def update_bonds(positions, orientations, types, bonds, particle_parameters, current_step, all_patch_data):
    """
    Updates the list of active bonds based on formation and breaking criteria.
    Supports both center-to-center and patch-based bonding.
    Uses particle_parameters for bonding criteria and patch definitions.

    Args:
        positions (np.ndarray): Current positions of particles.
        orientations (np.ndarray): Current orientations of particles.
        types (np.ndarray): Types of particles.
        bonds (dict): Current dictionary of active bonds.
        particle_parameters (dict): Dictionary of particle simulation parameters.
        current_step (int): The current simulation step number.
        all_patch_data (list of lists): Data structure containing info for all patches.

    Returns:
        dict: Updated dictionary of active bonds.
    """
    bonding_params = particle_parameters["bonding"]
    if not bonding_params["enabled"]:
        return bonds # Bonding is disabled

    formation_criteria = bonding_params["formation_criteria"]
    distance_tolerance = formation_criteria["distance_tolerance"]
    # Ensure patch_type_compatibility_matrix is a numpy array of bools
    patch_type_compatibility_matrix = np.array(formation_criteria.get("patch_type_compatibility_matrix", [[]]), dtype=bool)
    orientation_alignment_tolerance = formation_criteria["orientation_alignment_tolerance"]
    bond_break_distance = bonding_params["bond_break_distance"]
    # bond_types_param = tuple(bonding_params["bond_types"]) # Not directly used here, types in patches used

    patch_params = particle_parameters["forces"].get("patch_params", {})
    patch_enabled = patch_params.get("enabled", False)
    patch_definitions = patch_params.get("patch_definitions", {})

    updated_bonds = bonds.copy()

    num_particles = positions.shape[0]
    if num_particles == 0 or not all_patch_data : return updated_bonds # No particles or patch data means no bonds

    boundaries = particle_parameters["boundaries"]
    box_size = np.array([boundaries["x_max"] - boundaries["x_min"], boundaries["y_max"] - boundaries["y_min"]])

    # --- Bond Breaking ---
    bonds_to_break = []
    for bond_key in updated_bonds.keys():
         is_patch_bond = isinstance(bond_key, tuple) and len(bond_key) == 2 and isinstance(bond_key[0], tuple)
         if is_patch_bond:
              (i, p_idx_i), (j, p_idx_j) = bond_key
              # Check particles/patches exist and indices valid
              if not (i < num_particles and j < num_particles and i < len(all_patch_data) and j < len(all_patch_data) and \
                  p_idx_i < len(all_patch_data[i]) and p_idx_j < len(all_patch_data[j])):
                   bonds_to_break.append(bond_key); continue

              patch_i_pos = all_patch_data[i][p_idx_i]["position"]
              patch_j_pos = all_patch_data[j][p_idx_j]["position"]
              r_patch_raw = patch_i_pos - patch_j_pos
              r_patch_pbc = r_patch_raw - box_size * np.round(r_patch_raw / box_size)
              if np.linalg.norm(r_patch_pbc) > bond_break_distance:
                   bonds_to_break.append(bond_key)
         # else: # Center-center bonds are not formed by default in this configuration, but can be added
             # i,j = bond_key
             # if not (i < num_particles and j < num_particles): bonds_to_break.append(bond_key); continue
             # r_cc_raw = positions[j,:] - positions[i,:]
             # r_cc_pbc = r_cc_raw - box_size * np.round(r_cc_raw / box_size)
             # if np.linalg.norm(r_cc_pbc) > bonding_params.get("bond_break_distance", 2.0): # Use specific param for CC if needed
             #      bonds_to_break.append(bond_key)

    for bond_key in bonds_to_break:
        if bond_key in updated_bonds: del updated_bonds[bond_key]


    # --- Bond Formation (Patch-Based) ---
    if patch_enabled and patch_definitions:
        # Determine search cutoff based on potential range or break distance
        formation_search_cutoff = bond_break_distance # Max distance to consider formation
        max_patch_extension = 0
        for p_type, p_defs in patch_definitions.items():
            for p_spec in p_defs: max_patch_extension = max(max_patch_extension, p_spec.get("distance",0))


        for i in range(num_particles):
            for j in range(i + 1, num_particles):
                if not all_patch_data[i] or not all_patch_data[j]: continue

                r_ij_center_vec = positions[i, :] - positions[j, :]
                r_ij_center_vec_pbc = r_ij_center_vec - box_size * np.round(r_ij_center_vec / box_size)
                if np.linalg.norm(r_ij_center_vec_pbc) > formation_search_cutoff + 2 * max_patch_extension: continue


                for p_idx_i, patch_i_data in enumerate(all_patch_data[i]):
                    for p_idx_j, patch_j_data in enumerate(all_patch_data[j]):
                        potential_bond_key = tuple(sorted(((i, p_idx_i), (j, p_idx_j))))
                        if potential_bond_key in updated_bonds: continue

                        patch_i_pos, patch_j_pos = patch_i_data["position"], patch_j_data["position"]
                        patch_i_type, patch_j_type = patch_i_data["patch_type"], patch_j_data["patch_type"]

                        # Type compatibility check
                        if not (0 <= patch_i_type < patch_type_compatibility_matrix.shape[0] and \
                                0 <= patch_j_type < patch_type_compatibility_matrix.shape[1] and \
                                patch_type_compatibility_matrix[patch_i_type, patch_j_type]):
                             continue

                        r_patch_raw = patch_i_pos - patch_j_pos
                        r_patch_pbc = r_patch_raw - box_size * np.round(r_patch_raw / box_size)
                        r_patch_mag = np.linalg.norm(r_patch_pbc)

                        # Distance criteria for formation
                        # Use characteristic distance from bond potential type
                        char_bond_dist = bonding_params.get("patch_bond_potential", {}).get("harmonic", {}).get("bond_distance", bond_break_distance)
                        if bonding_params.get("patch_bond_potential",{}).get("type") == "lennard_jones":
                            char_bond_dist = bonding_params.get("patch_bond_potential",{}).get("lennard_jones",{}).get("sigma", bond_break_distance)
                        elif bonding_params.get("patch_bond_potential",{}).get("type") == "square_well":
                            char_bond_dist = bonding_params.get("patch_bond_potential",{}).get("square_well",{}).get("sigma", bond_break_distance)


                        form_dist_met = False
                        if r_patch_mag < bond_break_distance: # Must be within break distance
                             if abs(r_patch_mag - char_bond_dist) < distance_tolerance : form_dist_met = True
                             if bonding_params.get("patch_bond_potential",{}).get("type") == "square_well":
                                 sw_sig = bonding_params.get("patch_bond_potential",{}).get("square_well",{}).get("sigma", bond_break_distance)
                                 sw_lam = bonding_params.get("patch_bond_potential",{}).get("square_well",{}).get("lambda", 1.0)
                                 if sw_sig <= r_patch_mag < sw_sig * sw_lam : form_dist_met = True


                        if form_dist_met:
                            # Orientation alignment (if enabled)
                            orient_align_met = True
                            if orientation_alignment_tolerance is not None and orientation_alignment_tolerance < np.pi: # Check if non-trivial tolerance
                                ideal_angle_map = particle_parameters["forces"]["orientation_potential"]["bond_angle_potential"].get("ideal_angle_mapping", {})

                                # Patch i orientation
                                particle_type_i_int = types[i]
                                # Defensive initialization for pdefs_i
                                pdefs_i = []
                                if particle_type_i_int in patch_definitions:
                                     pdefs_i = patch_definitions.get(particle_type_i_int)
                                elif str(particle_type_i_int) in patch_definitions: # Fallback to string key if int key not found
                                     pdefs_i = patch_definitions.get(str(particle_type_i_int))
                                # Ensure it is a list
                                if not isinstance(pdefs_i, list): pdefs_i = []

                                if p_idx_i >= len(pdefs_i): continue
                                angle_rel_i = pdefs_i[p_idx_i].get("angle_relative_to_particle",0.0)
                                abs_angle_i = orientations[i] + angle_rel_i
                                patch_i_orient_vec = np.array([np.cos(abs_angle_i), np.sin(abs_angle_i)])
                                ideal_i = ideal_angle_map.get(patch_i_type, 0.0) # ideal_angle_map has int keys
                                angle_i_to_bond = np.arctan2(patch_i_orient_vec[0]*(r_patch_pbc[1]/r_patch_mag) - patch_i_orient_vec[1]*(r_patch_pbc[0]/r_patch_mag), np.dot(patch_i_orient_vec, r_patch_pbc/r_patch_mag))
                                dev_i = abs(np.mod(angle_i_to_bond - ideal_i + np.pi, 2*np.pi) - np.pi)


                                # Patch j orientation
                                particle_type_j_int = types[j]
                                # Defensive initialization for pdefs_j
                                pdefs_j = []
                                if particle_type_j_int in patch_definitions:
                                     pdefs_j = patch_definitions.get(particle_type_j_int)
                                elif str(particle_type_j_int) in patch_definitions: # Fallback to string key
                                     pdefs_j = patch_definitions.get(str(particle_type_j_int))
                                # Ensure it is a list
                                if not isinstance(pdefs_j, list): pdefs_j = []

                                if p_idx_j >= len(pdefs_j): continue
                                angle_rel_j = pdefs_j[p_idx_j].get("angle_relative_to_particle",0.0)
                                abs_angle_j = orientations[j] + angle_rel_j
                                patch_j_orient_vec = np.array([np.cos(abs_angle_j), np.sin(abs_angle_j)])
                                ideal_j = ideal_angle_map.get(patch_j_type, 0.0)
                                angle_j_to_bond = np.arctan2(patch_j_orient_vec[0]*(-bond_vec_dir[1])-patch_j_orient_vec[1]*(-bond_vec_dir[0]), np.dot(patch_j_orient_vec,-bond_vec_dir))
                                dev_j = abs(np.mod(angle_j_to_bond - ideal_j + np.pi, 2*np.pi) - np.pi)


                                if dev_i > orientation_alignment_tolerance or dev_j > orientation_alignment_tolerance:
                                     orient_align_met = False

                            if orient_align_met:
                                 # Form bond
                                 initial_strength = bonding_params.get("patch_bond_potential",{}).get("harmonic",{}).get("bond_strength", 100.0) # Store initial harmonic for adaptive
                                 updated_bonds[potential_bond_key] = {'formed_step': current_step, 'initial_strength': initial_strength, 'patch_pair': (p_idx_i, p_idx_j)}

    return updated_bonds


# Function for Particle Creation (remains the same, uses particle_parameters)
def create_particles(positions, velocities, orientations, types, masses, bonds, particle_parameters, current_step, all_patch_data):
    """
    Handles particle creation based on defined criteria.
    (Placeholder - actual creation logic needs to be implemented based on triggers,
    possibly influenced by brain state).
    Uses particle_parameters for creation criteria and new particle properties.
    """
    creation_params = particle_parameters.get("particle_creation", {})
    if not creation_params.get("enabled", False) or random.random() > creation_params.get("creation_rate",0.0): # Stochastic rate
        return positions, velocities, orientations, types, masses, bonds, all_patch_data, len(positions)

    num_existing_particles = len(positions)
    new_particle_config = creation_params.get("new_particle", {})
    trigger_config = creation_params.get("trigger", {})
    # Placeholder for trigger logic (e.g. a specific particle type initiating creation)
    # For simplicity, let's try to add one particle if any particle exists
    if num_existing_particles == 0:
        return positions, velocities, orientations, types, masses, bonds, all_patch_data, num_existing_particles


    # Select a random existing particle as parent (very simplified trigger)
    parent_idx = random.randrange(num_existing_particles)
    parent_pos = positions[parent_idx]
    parent_vel = velocities[parent_idx]
    parent_orient = orientations[parent_idx]
    # parent_ang_vel = angular_velocities[parent_idx] # If angular_velocities available


    new_type = new_particle_config.get("type", 0)
    new_pos_offset = np.array([ (np.random.rand()-0.5)*2 , (np.random.rand()-0.5)*2 ]) # Random offset
    new_pos = parent_pos + new_pos_offset
    new_vel = parent_vel * new_particle_config.get("initial_velocity_scale", 0.1)

    new_orient_config = new_particle_config.get("angular_initialization", particle_parameters["initial_conditions"]["new_particle_angular_initialization"])
    new_orient = parent_orient if new_orient_config["orientation_type"] == "copy_parent" else new_orient_config["orientation_angle"]
    # new_ang_vel = parent_ang_vel * new_orient_config["angular_velocity_scale"] if new_orient_config["angular_velocity_type"] == "copy_scaled_parent" else 0.0


    new_mass = particle_parameters["initial_conditions"]["mass_mapping"].get(new_type, 1.0)

    # Append new particle
    positions = np.vstack([positions, new_pos])
    velocities = np.vstack([velocities, new_vel])
    orientations = np.append(orientations, new_orient)
    types = np.append(types, new_type)
    masses = np.append(masses, new_mass)
    # angular_velocities = np.append(angular_velocities, new_ang_vel) # If tracking
    # accelerations & angular_accelerations will need to be resized too

    # Update all_patch_data: add empty list for the new particle
    all_patch_data.append([]) # This will be properly populated in next force calculation

    num_particles = len(positions)
    # print(f"Particle created. Total particles: {num_particles}")
    return positions, velocities, orientations, types, masses, bonds, all_patch_data, num_particles


# Function for Particle Deletion (remains the same, uses particle_parameters)
def delete_particles(positions, velocities, orientations, types, masses, bonds, particle_parameters, current_step, all_patch_data):
    """
    Handles particle deletion based on defined criteria.
    (Placeholder - actual deletion logic needs to be implemented based on triggers,
    possibly influenced by brain state).
    Uses particle_parameters for deletion criteria.
    """
    deletion_params = particle_parameters.get("particle_deletion", {})
    if not deletion_params.get("enabled", False) or random.random() > deletion_params.get("deletion_rate",0.0):
        return positions, velocities, orientations, types, masses, bonds, all_patch_data, len(positions)

    num_particles_before_del = len(positions)
    if num_particles_before_del == 0: return positions, velocities, orientations, types, masses, bonds, all_patch_data, 0


    trigger = deletion_params.get("trigger", {})
    condition = trigger.get("condition", None)
    buffer_distance = trigger.get("buffer_distance", 0.0)
    particles_to_delete_indices = []

    if condition == "out_of_bounds":
         boundaries = particle_parameters["boundaries"]
         x_min, x_max = boundaries["x_min"], boundaries["x_max"]
         y_min, y_max = boundaries["y_min"], boundaries["y_max"]
         for i in range(num_particles_before_del):
              if (positions[i, 0] < x_min - buffer_distance or positions[i, 0] > x_max + buffer_distance or
                  positions[i, 1] < y_min - buffer_distance or positions[i, 1] > y_max + buffer_distance):
                   particles_to_delete_indices.append(i)
    # Add other deletion conditions if needed

    if particles_to_delete_indices:
         keep_mask = np.ones(num_particles_before_del, dtype=bool)
         keep_mask[particles_to_delete_indices] = False

         positions = positions[keep_mask]
         velocities = velocities[keep_mask]
         orientations = orientations[keep_mask]
         types = types[keep_mask]
         masses = masses[keep_mask]
         # Resize accelerations if they are managed outside:
         # accelerations = accelerations[keep_mask]
         # angular_velocities = angular_velocities[keep_mask]
         # angular_accelerations = angular_accelerations[keep_mask]

         all_patch_data_updated = [patch_list for i, patch_list in enumerate(all_patch_data) if keep_mask[i]]
         all_patch_data = all_patch_data_updated

         # Update bonds: remove bonds involving deleted particles and re-index remaining
         old_to_new_map = {old_idx: new_idx for new_idx, old_idx in enumerate(np.where(keep_mask)[0])}
         updated_bonds = {}
         for bond_key, bond_info in bonds.items():
             is_patch_bond = isinstance(bond_key, tuple) and len(bond_key) == 2 and isinstance(bond_key[0], tuple)
             if is_patch_bond:
                  (i, p_idx_i), (j, p_idx_j) = bond_key
                  if i in old_to_new_map and j in old_to_new_map:
                       updated_bonds[tuple(sorted(((old_to_new_map[i], p_idx_i), (old_to_new_map[j], p_idx_j))))] = bond_info
             # else: # Center-center bond
                 # i,j = bond_key
                 # if i in old_to_new_map and j in old_to_new_map:
                 #      updated_bonds[tuple(sorted((old_to_new_map[i], old_to_new_map[j])))] = bond_info
         bonds = updated_bonds
         # print(f"Deleted {len(particles_to_delete_indices)} particles. Remaining: {len(positions)}")


    return positions, velocities, orientations, types, masses, bonds, all_patch_data, len(positions)


# Function for State Change on Bond Formation (remains the same, uses particle_parameters)
def apply_state_change_on_bond_form(types, bonds, particle_parameters, current_step):
    """
    Changes the type of a particle when a specific bond is formed.
    (Placeholder - actual logic needs to be implemented based on triggers).
    Uses particle_parameters for state change criteria.
    """
    state_change_params = particle_parameters.get("state_change", {})
    if not state_change_params.get("enabled", False):
        return types

    on_bond_form_params = state_change_params.get("on_bond_form", {})
    from_type_config = on_bond_form_params.get("from_type", None) # This could be an int
    to_type_config = on_bond_form_params.get("to_type", None) # This could be an int

    if from_type_config is None or to_type_config is None:
         return types

    updated_types = np.copy(types)
    num_particles = updated_types.shape[0]

    for bond_key, bond_info in bonds.items():
         if bond_info.get('formed_step', -1) == current_step: # Bond formed in current step
              is_patch_bond = isinstance(bond_key, tuple) and len(bond_key) == 2 and isinstance(bond_key[0], tuple)
              if is_patch_bond:
                   (i, _), (j, _) = bond_key # Don't need patch indices for type change of particle
                   if i < num_particles and updated_types[i] == from_type_config: updated_types[i] = to_type_config
                   if j < num_particles and updated_types[j] == from_type_config: updated_types[j] = to_type_config
              # else: # Center-center bond
                   # i,j = bond_key
                   # if i < num_particles and updated_types[i] == from_type_config : updated_types[i] = to_type_config
                   # if j < num_particles and updated_types[j] == from_type_config : updated_types[j] = to_type_config
    return updated_types


# Function to Get Cluster Labels (remains the same)
def get_cluster_labels_for_frame(positions, bonds, num_particles):
    """
    Identifies clusters of particles based on active bonds.
    """
    if num_particles == 0:
        return np.array([], dtype=int)

    graph = scipy.sparse.lil_matrix((num_particles, num_particles), dtype=int)
    for bond_key in bonds.keys():
         is_patch_bond = isinstance(bond_key, tuple) and len(bond_key) == 2 and isinstance(bond_key[0], tuple)
         if is_patch_bond:
              (i, _), (j, _) = bond_key # Particle indices are i and j
              if i < num_particles and j < num_particles:
                   graph[i, j] = 1; graph[j, i] = 1
         # else: # Center-center bond
             # i, j = bond_key
             # if i < num_particles and j < num_particles:
             #      graph[i, j] = 1; graph[j, i] = 1

    n_components, labels = scipy.sparse.csgraph.connected_components(graph, directed=False, connection='weak') # Weak for undirected
    return labels


# Function to Save Simulation State (remains the same, uses particle_parameters)
def save_simulation_state(step, positions, velocities, orientations, types, masses, bonds, particle_parameters):
    """
    Saves the current state of the particle simulation to files.
    Uses particle_parameters for the save directory.
    """
    save_directory = particle_parameters["saving"]["save_directory"]
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Convert bonds to JSON-serializable format (list of lists for keys, info as dict)
    serializable_bonds = {}
    for k, v in bonds.items():
        # Key: Convert ((p1_idx, patch1_idx), (p2_idx, patch2_idx)) to string "p1_patch1-p2_patch2"
        # Or handle simpler int-tuple keys for center-center bonds.
        # For pickle, tuples are fine as keys. If this was for JSON, more care needed.
        # Since it's pickle, we can store bonds dictionary directly.
        serializable_bonds[k] = v


    state = {
        "step": step,
        "positions": positions.tolist(),
        "velocities": velocities.tolist(),
        "orientations": orientations.tolist(),
        "types": types.tolist(),
        "masses": masses.tolist(),
        "bonds": serializable_bonds, # Store the dict as-is for pickle
        "parameters": particle_parameters
    }

    filepath = os.path.join(save_directory, f"particle_state_step_{step:06d}.pkl")
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    except Exception as e:
        print(f"Error saving particle simulation state to {filepath}: {e}")


# Function to Load Simulation State (remains the same, uses particle_parameters)
def load_simulation_state(particle_parameters_input): # Rename to avoid conflict with global
    """
    Loads the latest particle simulation state from files in the load directory.
    Uses particle_parameters_input for the load directory.
    """
    load_directory = particle_parameters_input["saving"]["load_directory"]
    if not os.path.exists(load_directory):
        print(f"Particle load directory '{load_directory}' not found. Starting from initial conditions.")
        return None

    state_files = [f for f in os.listdir(load_directory) if f.startswith("particle_state_step_") and f.endswith(".pkl")]
    if not state_files:
        print(f"No particle simulation state files found in '{load_directory}'. Starting from initial conditions.")
        return None

    try: # Add try block for parsing step numbers
        latest_step = max([int(f.split('_')[-1].split('.')[0]) for f in state_files])
    except ValueError:
        print(f"Error parsing step numbers from filenames in '{load_directory}'. Starting fresh.")
        return None

    latest_filepath = os.path.join(load_directory, f"particle_state_step_{latest_step:06d}.pkl")

    try:
        with open(latest_filepath, 'rb') as f:
            state = pickle.load(f)
        print(f"Particle simulation state loaded successfully from '{latest_filepath}'")

        # Use parameters from the loaded state primarily
        loaded_parameters = state.get("parameters", particle_parameters_input)
        # Override with current script's visualization params if desired, or merge carefully
        # For now, strictly use loaded_parameters for consistency of the loaded state.

        loaded_types_np = np.array(state["types"], dtype=int)
        vis_params = loaded_parameters.get("visualization", {})
        patches_vis_params = vis_params.get("patches", {})
        color_mapping_from_loaded = patches_vis_params.get("color_mapping", {})
        # Ensure color_mapping keys are int for lookup, as types are int
        color_mapping_int_keys = {int(k):v for k,v in color_mapping_from_loaded.items()} if isinstance(color_mapping_from_loaded,dict) else {}


        loaded_colors = [color_mapping_int_keys.get(t, 'gray') for t in loaded_types_np]


        # Accelerations and angular velocities/accelerations are re-calculated at simulation start.
        num_loaded_particles = len(state["positions"])
        return (
            state["step"],
            np.array(state["positions"]),
            np.array(state["velocities"]),
            np.zeros((num_loaded_particles,2)) if num_loaded_particles > 0 else np.array([]).reshape(0,2), # Accelerations placeholder
            state.get("bonds", {}), # Bonds directly from pickle
            loaded_types_np,
            np.array(state["masses"]),
            loaded_colors,
            np.array(state["orientations"]),
            np.zeros(num_loaded_particles) if num_loaded_particles > 0 else np.array([]), # Angular velocities placeholder
            np.zeros(num_loaded_particles) if num_loaded_particles > 0 else np.array([]), # Angular accelerations placeholder
            num_loaded_particles,
            loaded_parameters, # Return the parameters that were part of the loaded state
            [] # Placeholder for all_patch_data, regenerated in first step
        )

    except FileNotFoundError:
        print(f"Error loading particle state from '{latest_filepath}'. File not found.")
        return None
    except Exception as e:
        print(f"An error occurred loading particle simulation state: {e}. Starting from initial conditions.")
        return None


# Analysis Function: Radial Distribution Function (RDF) (remains the same, uses particle_parameters)
def calculate_radial_distribution_function(positions_history, particle_parameters):
    """
    Calculates the radial distribution function (g(r)) for the particle system.
    Calculates g(r) averaged over specified frames.
    Uses particle_parameters for analysis settings and boundaries.
    Requires positions_history.
    """
    analysis_params = particle_parameters["analysis"]
    rdf_dr = analysis_params.get("rdf_dr", 0.1)
    boundaries = particle_parameters["boundaries"] # Get boundaries from particle_parameters
    rdf_rmax = analysis_params.get("rdf_rmax", (boundaries["x_max"] - boundaries["x_min"]) / 2.0)
    rdf_start_frame = analysis_params.get("rdf_start_frame", 0)


    if not positions_history or len(positions_history) <= rdf_start_frame:
        # print("Warning: No particle position data available for RDF or start frame out of bounds.")
        return np.array([]), np.array([])


    box_size = np.array([boundaries["x_max"] - boundaries["x_min"], boundaries["y_max"] - boundaries["y_min"]])
    if box_size[0] <=0 or box_size[1] <=0 : return np.array([]), np.array([]) # Invalid box


    num_bins = int(rdf_rmax / rdf_dr)
    if num_bins <=0 : return np.array([]), np.array([])
    distance_counts = np.zeros(num_bins)
    bin_edges = np.linspace(0, rdf_rmax, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total_number_density_sum = 0.0 # Sum of number densities over frames
    num_frames_averaged = 0
    total_particles_in_avg_frames = 0 # Sum of N over averaged frames


    for frame_index in range(rdf_start_frame, len(positions_history)):
        positions = positions_history[frame_index]
        if not isinstance(positions, np.ndarray) or positions.ndim != 2 or positions.shape[1]!=2: continue # Invalid position data for frame
        num_particles_in_frame = positions.shape[0]
        if num_particles_in_frame < 2: continue

        num_frames_averaged += 1
        total_particles_in_avg_frames += num_particles_in_frame


        r_vec_all_pairs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        r_vec_all_pairs_pbc = r_vec_all_pairs - box_size * np.round(r_vec_all_pairs / box_size)
        r_mag_all_pairs = np.sqrt(np.sum(r_vec_all_pairs_pbc**2, axis=2))

        upper_triangle_indices = np.triu_indices_from(r_mag_all_pairs, k=1)
        distances = r_mag_all_pairs[upper_triangle_indices]
        distances_within_rmax = distances[distances < rdf_rmax]

        valid_distances = distances_within_rmax[np.isfinite(distances_within_rmax)]
        if valid_distances.size > 0:
             counts, _ = np.histogram(valid_distances, bins=bin_edges)
             distance_counts += counts


        area = box_size[0] * box_size[1]
        if area > DEFAULT_EPSILON: # Ensure area is positive
             total_number_density_sum += num_particles_in_frame / area


    if num_frames_averaged == 0 or np.sum(distance_counts) == 0:
         # print("Warning: No valid frames or pairs found for RDF calculation.")
         return np.array([]), np.array([])

    avg_num_particles_for_norm = total_particles_in_avg_frames / num_frames_averaged
    avg_number_density_for_norm = total_number_density_sum / num_frames_averaged


    rdf_g_r = np.zeros(num_bins)
    for i in range(num_bins):
        r_inner, r_outer = bin_edges[i], bin_edges[i+1]
        area_of_bin = np.pi * (r_outer**2 - r_inner**2)
        # Expected number of pairs in this bin FOR ONE FRAME, for N particles, with density rho:
        # N * rho * area_bin / 2. We sum counts over all frames, so multiply by num_frames_averaged.
        expected_pairs_in_bin_total = num_frames_averaged * avg_num_particles_for_norm * avg_number_density_for_norm * area_of_bin / 2.0

        if expected_pairs_in_bin_total > DEFAULT_EPSILON:
             rdf_g_r[i] = distance_counts[i] / expected_pairs_in_bin_total
        else: rdf_g_r[i] = 0.0

    return bin_centers, rdf_g_r


# Plotting Function for RDF (remains the same, uses particle_parameters)
def plot_rdf(rdf_bin_centers, rdf_g_r, particle_parameters):
    if rdf_bin_centers.size == 0 or rdf_g_r.size == 0: return # Check if arrays are empty
    save_directory = particle_parameters["saving"]["save_directory"]
    if not os.path.exists(save_directory): os.makedirs(save_directory)
    plt.figure(); plt.plot(rdf_bin_centers, rdf_g_r); plt.xlabel("Distance (r)")
    plt.ylabel("g(r)"); plt.title("Particle Radial Distribution Function"); plt.grid(True)
    ymin = 0.0; ymax = max(2.0, np.max(rdf_g_r[np.isfinite(rdf_g_r)]) * 1.1 if np.any(np.isfinite(rdf_g_r)) else 2.0)
    plt.ylim(ymin, ymax)
    plt.savefig(os.path.join(save_directory, "particle_rdf_plot.png")); plt.close()
    print(f"Particle RDF plot saved to {os.path.join(save_directory, 'particle_rdf_plot.png')}")


# Analysis Function: Mean Squared Displacement (MSD) (remains the same, uses particle_parameters)
def calculate_mean_squared_displacement(positions_history, final_types, particle_parameters):
    analysis_params = particle_parameters["analysis"]
    msd_start_frame = analysis_params.get("msd_start_frame", analysis_params.get("rdf_start_frame", 0))

    if not positions_history or len(positions_history) <= msd_start_frame +1: # Need at least 2 frames from start
        # print("Warning: Insufficient position data for MSD.")
        return np.array([]), np.array([]), {}

    num_total_frames = len(positions_history)
    boundaries = particle_parameters["boundaries"]
    box_size = np.array([boundaries["x_max"] - boundaries["x_min"], boundaries["y_max"] - boundaries["y_min"]])
    dt_sim = particle_parameters["simulation"]["dt"]


    # Time lags to calculate MSD for (up to half the duration of analyzed trajectory segment)
    max_lag_steps = (num_total_frames - msd_start_frame) // 2
    if max_lag_steps <=0: return np.array([]), np.array([]), {} # Not enough steps for any lag
    msd_time_lags_dt = np.arange(1, max_lag_steps + 1) # Time lags in units of steps
    msd_time_points_sec = msd_time_lags_dt * dt_sim


    overall_msd = np.zeros(max_lag_steps)
    msd_per_type = collections.defaultdict(lambda: np.zeros(max_lag_steps))
    # Counts for averaging, ensure final_types is an np.array
    final_types_np = np.array(final_types) if not isinstance(final_types, np.ndarray) else final_types


    for lag_idx, time_lag_steps in enumerate(msd_time_lags_dt):
        squared_displacements_for_lag = []
        type_specific_sq_disp_for_lag = collections.defaultdict(list)

        # Iterate over possible start times (origins) for this lag
        for origin_frame_idx in range(msd_start_frame, num_total_frames - time_lag_steps):
            pos_origin = positions_history[origin_frame_idx]
            pos_lagged = positions_history[origin_frame_idx + time_lag_steps]

            # Ensure particle count consistency for this specific pair of frames
            if pos_origin.shape[0] != pos_lagged.shape[0] or pos_origin.shape[0] == 0:
                continue # Skip if particle numbers differ or no particles

            # Assume final_types corresponds to particles at origin_frame_idx
            # This is an approximation if particles are created/deleted
            current_types_for_origin = final_types_np
            if len(final_types_np) != pos_origin.shape[0]:
                 # If final_types doesn't match, try to get types from types_history (if available and passed)
                 # For simplicity here, we proceed with warning or skip type-specific
                 # print(f"Warning: final_types length mismatch for MSD at frame {origin_frame_idx}. Type-specific MSD may be affected.")
                 # If types_history was passed:
                 # current_types_for_origin = types_history[origin_frame_idx] if origin_frame_idx < len(types_history) else final_types_np
                 pass


            disp_raw = pos_lagged - pos_origin
            disp_pbc = disp_raw - box_size * np.round(disp_raw / box_size)
            sq_disp_per_particle = np.sum(disp_pbc**2, axis=1)

            squared_displacements_for_lag.extend(sq_disp_per_particle)
            if len(current_types_for_origin) == len(sq_disp_per_particle): # If types array matches
                 for p_idx, sq_d in enumerate(sq_disp_per_particle):
                      type_specific_sq_disp_for_lag[current_types_for_origin[p_idx]].append(sq_d)


        if squared_displacements_for_lag:
            overall_msd[lag_idx] = np.mean(squared_displacements_for_lag)
        for p_type, disp_list in type_specific_sq_disp_for_lag.items():
            if disp_list: msd_per_type[p_type][lag_idx] = np.mean(disp_list)

    return msd_time_points_sec, overall_msd, dict(msd_per_type) # Convert defaultdict to dict


# Plotting Function for MSD (remains the same, uses particle_parameters)
def plot_msd(msd_time_points, overall_msd, type_msd_dict, particle_parameters):
    if msd_time_points.size == 0 or overall_msd.size == 0: return
    save_directory = particle_parameters["saving"]["save_directory"]
    if not os.path.exists(save_directory): os.makedirs(save_directory)

    plt.figure(); plt.plot(msd_time_points, overall_msd, label="Overall MSD")
    color_map_vis = particle_parameters.get("visualization", {}).get("patches", {}).get("color_mapping", {})
    # Ensure color_map_vis keys are int if types are int
    color_map_int_keys = {int(k):v for k,v in color_map_vis.items()} if isinstance(color_map_vis,dict) else {}


    for p_type, msd_arr in type_msd_dict.items():
        if msd_arr.size == msd_time_points.size:
            plt.plot(msd_time_points, msd_arr, label=f"Type {p_type} MSD", color=color_map_int_keys.get(p_type, 'gray'))
    plt.xlabel("Time (s)"); plt.ylabel("MSD (distance$^2$)")
    plt.title("Particle Mean Squared Displacement"); plt.legend(); plt.grid(True); plt.xscale('log'); plt.yscale('log') # Often plotted log-log
    plt.savefig(os.path.join(save_directory, "particle_msd_plot.png")); plt.close()
    print(f"Particle MSD plot saved to {os.path.join(save_directory, 'particle_msd_plot.png')}")


# Analysis Function: Bond Angle Distribution (remains the same, uses particle_parameters)
def calculate_bond_angle_distribution(positions_history, orientations_history, all_patch_data_history, bonds_history, types_history, particle_parameters):
    orient_pot_params = particle_parameters["forces"].get("orientation_potential", {})
    bond_angle_pot_params = orient_pot_params.get("bond_angle_potential", {})
    if not bond_angle_pot_params.get("enabled", False): return np.array([]), np.array([])

    patch_defs = particle_parameters["forces"].get("patch_params", {}).get("patch_definitions", {})
    if not patch_defs: return np.array([]), np.array([])

    ideal_angle_map = bond_angle_pot_params.get("ideal_angle_mapping", {}) # Keys are int

    angles_dev = []
    boundaries = particle_parameters["boundaries"]
    box_size = np.array([boundaries["x_max"] - boundaries["x_min"], boundaries["y_max"] - boundaries["y_min"]])

    min_history_len = min(len(h) for h in [positions_history, orientations_history, all_patch_data_history, bonds_history, types_history])

    for frame_idx in range(min_history_len):
        bonds_f, pos_f, orient_f, patches_f, types_f = bonds_history[frame_idx], positions_history[frame_idx], orientations_history[frame_idx], all_patch_data_history[frame_idx], types_history[frame_idx]
        num_p_f = pos_f.shape[0]

        for bond_key in bonds_f.keys():
            if not (isinstance(bond_key, tuple) and len(bond_key) == 2 and isinstance(bond_key[0], tuple)): continue
            (i, p_idx_i), (j, p_idx_j) = bond_key
            if not (i<num_p_f and j<num_p_f and i<len(patches_f) and j<len(patches_f) and p_idx_i<len(patches_f[i]) and p_idx_j<len(patches_f[j])): continue

            patch_i_dat, patch_j_dat = patches_f[i][p_idx_i], patches_f[j][p_idx_j]
            patch_i_type, patch_j_type = patch_i_dat["patch_type"], patch_j_dat["patch_type"]

            r_patch_raw = patch_j_dat["position"] - patch_i_dat["position"]
            r_patch_pbc = r_patch_raw - box_size * np.round(r_patch_raw / box_size)
            r_patch_mag = np.linalg.norm(r_patch_pbc)
            if r_patch_mag < DEFAULT_EPSILON: continue
            bond_vec_dir = r_patch_pbc / r_patch_mag

            particle_type_i_int = types_f[i]
            # Defensive initialization for pdefs_i
            pdefs_i = []
            if particle_type_i_int in patch_defs:
                pdefs_i = patch_defs.get(particle_type_i_int)
            elif str(particle_type_i_int) in patch_defs:
                pdefs_i = patch_defs.get(str(particle_type_i_int))
            if not isinstance(pdefs_i, list): pdefs_i = [] # Final safeguard

            if p_idx_i >= len(pdefs_i): continue
            angle_rel_i = pdefs_i[p_idx_i].get("angle_relative_to_particle",0.0)
            abs_angle_i = orient_f[i] + angle_rel_i
            patch_i_orient_vec = np.array([np.cos(abs_angle_i), np.sin(abs_angle_i)])
            ideal_i_rad = ideal_angle_map.get(patch_i_type, 0.0) # patch_i_type is int
            actual_i_angle = np.arctan2(patch_i_orient_vec[0]*bond_vec_dir[1]-patch_i_orient_vec[1]*bond_vec_dir[0], np.dot(patch_i_orient_vec,bond_vec_dir))
            angles_dev.append(abs(np.mod(actual_i_angle - ideal_i_rad + np.pi, 2*np.pi) - np.pi))


            particle_type_j_int = types_f[j]
            # Defensive initialization for pdefs_j
            pdefs_j = []
            if particle_type_j_int in patch_defs:
                pdefs_j = patch_defs.get(particle_type_j_int)
            elif str(particle_type_j_int) in patch_defs:
                pdefs_j = patch_defs.get(str(particle_type_j_int))
            if not isinstance(pdefs_j, list): pdefs_j = [] # Final safeguard

            if p_idx_j >= len(pdefs_j): continue
            angle_rel_j = pdefs_j[p_idx_j].get("angle_relative_to_particle",0.0)
            abs_angle_j = orient_f[j] + angle_rel_j
            patch_j_orient_vec = np.array([np.cos(abs_angle_j), np.sin(abs_angle_j)])
            ideal_j_rad = ideal_angle_map.get(patch_j_type, 0.0)
            actual_j_angle = np.arctan2(patch_j_orient_vec[0]*(-bond_vec_dir[1])-patch_j_orient_vec[1]*(-bond_vec_dir[0]), np.dot(patch_j_orient_vec,-bond_vec_dir))
            angles_dev.append(abs(np.mod(actual_j_angle - ideal_j_rad + np.pi, 2*np.pi) - np.pi))

    if not angles_dev: return np.array([]), np.array([])
    hist, edges = np.histogram(angles_dev, bins=50, range=(0,np.pi), density=True)
    centers = (edges[:-1] + edges[1:])/2
    return centers, hist


# Plotting Function for Bond Angle Distribution (remains the same, uses particle_parameters)
def plot_bond_angle_distribution(bin_centers, angle_counts, particle_parameters):
    if bin_centers.size == 0 or angle_counts.size == 0: return
    save_dir = particle_parameters["saving"]["save_directory"]
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    plt.figure(); plt.bar(bin_centers, angle_counts, width=(bin_centers[1]-bin_centers[0] if len(bin_centers)>1 else 0.1) , edgecolor='black')
    plt.xlabel("Angle Deviation (radians)"); plt.ylabel("Probability Density")
    plt.title("Patch Bond Angle Distribution"); plt.grid(axis='y')
    plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], ['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
    plt.savefig(os.path.join(save_dir, "particle_bond_angle_distribution.png")); plt.close()
    print(f"Particle bond angle distribution plot saved to {os.path.join(save_dir, 'particle_bond_angle_distribution.png')}")


# Analysis Function: Cluster Size Distribution (remains the same, uses particle_parameters)
def calculate_cluster_size_distribution(bonds_history, positions_history, particle_parameters):
    if not bonds_history or not positions_history : return np.array([]), np.array([])
    all_sizes = []
    min_len = min(len(bonds_history), len(positions_history))
    for frame_idx in range(min_len):
        bonds_f, pos_f = bonds_history[frame_idx], positions_history[frame_idx]
        num_p = pos_f.shape[0]
        if num_p > 0:
            labels = get_cluster_labels_for_frame(pos_f, bonds_f, num_p)
            _, counts = np.unique(labels, return_counts=True)
            all_sizes.extend(counts)
    if not all_sizes: return np.array([]), np.array([])
    max_size = max(all_sizes) if all_sizes else 1
    hist, edges = np.histogram(all_sizes, bins=np.arange(0.5, max_size+1.5,1.0), density=True)
    centers = np.arange(1, max_size+1)
    # Ensure hist matches centers length if max_size was small
    if len(hist) < len(centers) and len(centers) == 1 and max_size == 1: # common for single particles
        hist = np.array([1.0]) if sum(all_sizes)==len(all_sizes) and all_sizes[0]==1 else np.array([0.0])


    return centers, hist


# Plotting Function for Cluster Size Distribution (remains the same, uses particle_parameters)
def plot_cluster_size_distribution(cluster_sizes, cluster_counts, particle_parameters):
    if cluster_sizes.size == 0 or cluster_counts.size == 0: return
    save_dir = particle_parameters["saving"]["save_directory"]
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    plt.figure(); plt.bar(cluster_sizes, cluster_counts, width=1.0, edgecolor='black')
    plt.xlabel("Cluster Size"); plt.ylabel("Probability"); plt.title("Particle Cluster Size Distribution")
    plt.xticks(cluster_sizes) if len(cluster_sizes) < 20 else plt.xticks(np.arange(min(cluster_sizes),max(cluster_sizes)+1, max(1,int(len(cluster_sizes)/10)) ))

    plt.grid(axis='y'); plt.savefig(os.path.join(save_dir, "particle_cluster_size_distribution.png")); plt.close()
    print(f"Particle cluster size distribution plot saved to {os.path.join(save_dir, 'particle_cluster_size_distribution.png')}")


# Analysis Function: Calculate Nematic Order Parameter (remains the same)
def calculate_nematic_order_parameter(orientations):
    if len(orientations) == 0: return 0.0
    cos_2t, sin_2t = np.cos(2*orientations), np.sin(2*orientations)
    return np.sqrt(np.mean(cos_2t)**2 + np.mean(sin_2t)**2)


# Visualization Function (Adapted for Coupled Simulation)
def visualize_simulation(total_energy_history, positions_history, orientations_history, nematic_order_parameter_history, all_patch_data_history, bonds_history, types_history, particle_parameters):
    save_dir = particle_parameters["saving"]["save_directory"]
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    anim_file = os.path.join(save_dir, particle_parameters["saving"].get("animation_filename", "particle_sim.gif"))
    boundaries = particle_parameters["boundaries"]
    x_min, x_max, y_min, y_max = boundaries["x_min"], boundaries["x_max"], boundaries["y_min"], boundaries["y_max"]

    if total_energy_history:
        plt.figure(); plt.plot(total_energy_history); plt.xlabel("Step"); plt.ylabel("Total Energy")
        plt.title("Particle Total Energy"); plt.grid(True)
        plt.savefig(os.path.join(save_dir, "particle_total_energy_plot.png")); plt.close()
        print(f"Particle total energy plot saved to {os.path.join(save_dir, 'particle_total_energy_plot.png')}")

    if nematic_order_parameter_history:
        plt.figure(); plt.plot(nematic_order_parameter_history); plt.xlabel("Step"); plt.ylabel("Nematic Order (S)")
        plt.title("Particle Nematic Order Parameter"); plt.grid(True); plt.ylim(0, 1.1)
        plt.savefig(os.path.join(save_dir, "particle_nematic_order_parameter_plot.png")); plt.close()
        print(f"Particle nematic order plot saved to {os.path.join(save_dir, 'particle_nematic_order_parameter_plot.png')}")

    if not positions_history: print("No particle positions for animation."); return

    fig_anim, ax_anim = plt.subplots(); ax_anim.set_xlim(x_min, x_max); ax_anim.set_ylim(y_min, y_max)
    ax_anim.set_aspect('equal', adjustable='box'); ax_anim.set_title("Particle Simulation"); ax_anim.grid(True)

    vis_params = particle_parameters["visualization"]
    orient_line_p = vis_params.get("orientation_line", {})
    patches_vis_p = vis_params.get("patches", {})
    bonds_vis_p = vis_params.get("bonds", {})
    clusters_vis_p = vis_params.get("clusters", {})

    particle_scatter = ax_anim.scatter([], [], s=50) # Base size
    max_hist_particles = max(len(p) for p in positions_history if len(p)>0) if any(len(p)>0 for p in positions_history) else 0
    orientation_lines_plt = [ax_anim.plot([], [], color=orient_line_p.get("color", 'k'), lw=orient_line_p.get("linewidth", 1))[0] for _ in range(max_hist_particles)]
    patch_scatter_plt = ax_anim.scatter([], [], s=(patches_vis_p.get("size",0.5)**2)*20, edgecolors=patches_vis_p.get("edgecolor",'k'), zorder=3)
    bond_lines_plt = [] # To store line artists for bonds, cleared each frame
    time_text_plt = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes)

    min_hist_len_anim = min(len(h) for h in [positions_history, orientations_history, all_patch_data_history, bonds_history, types_history])


    def update_anim(frame):
        if frame >= min_hist_len_anim: return [particle_scatter, patch_scatter_plt, time_text_plt] + orientation_lines_plt + bond_lines_plt

        pos_f, orient_f, patches_f_all, bonds_f, types_f = positions_history[frame], orientations_history[frame], all_patch_data_history[frame], bonds_history[frame], types_history[frame]
        num_p_f = pos_f.shape[0]
        if num_p_f == 0: # Handle empty frame
             particle_scatter.set_offsets(np.array([]).reshape(0,2))
             patch_scatter_plt.set_offsets(np.array([]).reshape(0,2))
             for line in orientation_lines_plt: line.set_data([],[])
             for bline in bond_lines_plt: bline.remove()
             bond_lines_plt.clear()
             time_text_plt.set_text(f"Step: {frame}, Time: {frame*particle_parameters['simulation']['dt']:.1f} (No Particles)")
             return [particle_scatter, patch_scatter_plt, time_text_plt] + orientation_lines_plt

        particle_scatter.set_offsets(pos_f)
        # Coloring particles
        current_colors = ['gray'] * num_p_f # Default
        if clusters_vis_p.get("enabled", False):
            labels = get_cluster_labels_for_frame(pos_f, bonds_f, num_p_f)
            unique_labels = np.unique(labels)
            cmap_func = plt.get_cmap(clusters_vis_p.get("colormap", "viridis"))
            current_colors = [cmap_func(l/(len(unique_labels)-1)) if len(unique_labels)>1 else cmap_func(0.5) for l in labels]
        else:
            # color_map_main = vis_params.get("patches",{}).get("color_mapping",{}) # Use patch color map for main particle
            # Using the direct particle color mapping as derived in init
            main_color_mapping = patches_vis_p.get("color_mapping",{}) # int keys
            current_colors = [main_color_mapping.get(t, 'grey') for t in types_f]


        particle_scatter.set_facecolor(current_colors)


        for i in range(max_hist_particles):
            if i < num_p_f and orient_line_p.get("enabled", True):
                l = orient_line_p.get("length",1.0)
                end_pt = pos_f[i] + l * np.array([np.cos(orient_f[i]), np.sin(orient_f[i])])
                orientation_lines_plt[i].set_data([pos_f[i,0], end_pt[0]], [pos_f[i,1], end_pt[1]])
            else: orientation_lines_plt[i].set_data([],[])

        # Patches
        patch_coords, patch_colors_list = [], []
        if patches_vis_p.get("enabled", True) and frame < len(all_patch_data_history) and patches_f_all:
             patch_color_map = patches_vis_p.get("color_mapping",{}) # int keys
             for p_idx_particle in range(num_p_f): # Iterate up to current particles
                 if p_idx_particle < len(patches_f_all): # Check if patch data exists for this particle
                     for patch_data in patches_f_all[p_idx_particle]: # patches_f_all is list of lists
                         patch_coords.append(patch_data["position"])
                         patch_colors_list.append(patch_color_map.get(patch_data["patch_type"], 'black'))
        patch_scatter_plt.set_offsets(np.array(patch_coords) if patch_coords else np.array([]).reshape(0,2) )
        if patch_colors_list : patch_scatter_plt.set_facecolor(patch_colors_list)
        else: patch_scatter_plt.set_facecolor(np.array([]))



        for bline in bond_lines_plt: bline.remove()
        bond_lines_plt.clear()
        if bonds_vis_p.get("enabled", True) and bonds_f and frame < len(all_patch_data_history) and patches_f_all:
            for bond_key in bonds_f.keys():
                is_patch_bond = isinstance(bond_key, tuple) and len(bond_key)==2 and isinstance(bond_key[0],tuple)
                if is_patch_bond:
                    (i,pi), (j,pj) = bond_key
                    if i<num_p_f and j<num_p_f and i<len(patches_f_all) and j<len(patches_f_all) and pi<len(patches_f_all[i]) and pj<len(patches_f_all[j]):
                        pos1, pos2 = patches_f_all[i][pi]["position"], patches_f_all[j][pj]["position"]
                        line, = ax_anim.plot([pos1[0],pos2[0]], [pos1[1],pos2[1]],
                                             color=bonds_vis_p.get("color",'gray'),
                                             lw=bonds_vis_p.get("linewidth", 2.0),
                                             linestyle=bonds_vis_p.get("linestyle", '-'),
                                             zorder=0)
                        bond_lines_plt.append(line)
                # else: # Center-center bonds
                #     i,j = bond_key
                #     if i<num_p_f and j<num_p_f:
                #         line, = ax_anim.plot([pos_f[i,0],pos_f[j,0]], [pos_f[i,1],pos_f[j,1]], color=bonds_vis_p.get("color",'gray'), lw=bonds_vis_p.get("linewidth",1.5), zorder=0)
                #         bond_lines_plt.append(line)


        time_text_plt.set_text(f"Step: {frame}, Time: {frame*particle_parameters['simulation']['dt']:.1f}")
        return [particle_scatter, patch_scatter_plt, time_text_plt] + orientation_lines_plt + bond_lines_plt


    if min_hist_len_anim > 0 :
        ani = animation.FuncAnimation(fig_anim, update_anim, frames=min_hist_len_anim, blit=True, interval=1000/particle_parameters["simulation"]["animation_fps"], repeat=False)
        try:
            print(f"Saving animation to {anim_file}...")
            ani.save(anim_file, writer=animation.PillowWriter(fps=particle_parameters["simulation"]["animation_fps"]))
            print("Animation saved.")
        except Exception as e: print(f"Error saving animation: {e}. Ensure Pillow is installed.")
    plt.close(fig_anim)


# --- Enums, Constants, and Classes from MAUS Prototype ---
# (Adaptive Matrix Solver 0.1)

# --- Enumerations for Problem Types ---
class ProblemType(Enum):
    EIGENVALUE = 1
    SOLVE_LINEAR_SYSTEM = 2
    SVD = 3

# --- Global Configuration Parameters (Informing MAUS's Heuristics) ---
# These are adjustable hyperparameters that MAUS uses in its internal decision-making.
GLOBAL_DEFAULT_PSI_EPSILON_BASE = np.complex128(1e-20) # Base regularization magnitude (multiplied by aggression factor)
GLOBAL_DEFAULT_ALPHA_V_INITIAL = np.complex128(0.01) # Initial learning rate for candidates' steps
GLOBAL_MAX_PSI_ATTEMPTS = 25 # Max attempts for InverseIterateSolver per candidate update
GLOBAL_MAX_STUCK_FOR_RETIREMENT = 8 # Times a candidate can repeatedly fail before being retired (population management)
GLOBAL_MAX_STUCK_FOR_PRUNING = 5 # Used by the population manager, indicates `Fragile` state when avg stuckness is higher than this value
GLOBAL_MIN_WEIGHT_TO_SURVIVE_PRUNE = 1e-10 # Minimum confidence weight for a candidate to stay in population
GLOBAL_VECTOR_SIMILARITY_TOL = 0.999 # Cosine similarity threshold for considering two vectors "the same"
GLOBAL_LAMBDA_SIMILARITY_TOL = 1e-5 # Absolute difference for eigenvalue uniqueness
GLOBAL_SIGMA_SIMILARITY_TOL_ABS = 1e-6 # Absolute threshold for singular value uniqueness (below this, it's considered ~0)
GLOBAL_SIGMA_SIMILARITY_TOL_REL = 1e-4 # Relative threshold for singular value uniqueness (e.g., 1e-4 of max sigma)
GLOBAL_CONVERGENCE_RESIDUAL_TOL = 1e-8 # Default global residual tolerance for MAUS solve.


# --- InverseIterateSolver: Adaptive Local Solver for Ax=b-like Problems ---
# This class encapsulates the robust linear system solving using direct_solve or iterative_gmres,
# dynamically choosing or falling back, and applying Ψ regularization.
class InverseIterateSolver:
    def __init__(self, N, base_psi_epsilon, max_attempts, preferred_method='direct_solve', is_sparse=False):
        self.N = N # Dimension of the matrix for solve
        self.base_psi_epsilon = base_psi_epsilon # Base magnitude for Ψ
        self.max_attempts = max_attempts # Max internal retries for a single solve call
        self.preferred_method = preferred_method # 'direct_solve' (sla.solve or spsolve) or 'iterative_gmres'
        self.fallback_method = 'iterative_gmres' if preferred_method == 'direct_solve' else 'direct_solve' # Auto-determines fallback
        self.is_sparse = is_sparse # True if problem is sparse

    def solve(self, A_target, b_rhs, candidate_stuck_counter):
        """
        Attempts to solve A_target @ x = b_rhs robustly with Psi regularization,
        potentially trying fallback solvers.
        """
        num_psi_attempts = 0
        current_method_for_try = self.preferred_method # Start with preferred method

        while num_psi_attempts < self.max_attempts:
            # Scale PSI by base and attempt count for increasing aggression based on history
            psi_scalar_magnitude = self.base_psi_epsilon * (10**(num_psi_attempts / 2.0)) * (10**(candidate_stuck_counter / 3.0))

            # Create regularization term (Psi): dynamically chooses sparse identity or dense random matrix
            if self.is_sparse:
                regularization_term = sp.identity(self.N, dtype=A_target.dtype, format='csc') * psi_scalar_magnitude
                # Note: `A_target` should already be in a sparse format for addition.
            else: # Dense matrix: adds random noise component to Psi
                random_perturb = (np.random.rand(self.N, self.N) - 0.5 + 1j * (np.random.rand(self.N, self.N) - 0.5)) * psi_scalar_magnitude * 0.15
                regularization_term = psi_scalar_magnitude * np.eye(self.N, dtype=A_target.dtype) + random_perturb

            # Add regularization to the target matrix for solving
            H_solve = A_target + regularization_term

            try:
                # Core solving logic based on `current_method_for_try`
                if current_method_for_try == 'direct_solve':
                    if self.is_sparse:
                        result_vec = spla.spsolve(H_solve.tocsc(), b_rhs) # scipy.sparse.linalg.spsolve for sparse direct solve
                    else:
                        result_vec = sla.solve(H_solve, b_rhs, assume_a='general') # np.linalg.solve for dense direct solve

                elif current_method_for_try == 'iterative_gmres':
                    # GMRES (Generalized Minimal Residual): robust for non-symmetric systems, can handle near-singularity by finding least-squares sol.
                    # It accepts both dense NumPy arrays and sparse SciPy matrices.
                    # x0: initial guess for solution. tol: relative tolerance. maxiter: max iterations.
                    x0_init = b_rhs if b_rhs.shape == H_solve.shape[1:] else np.zeros_like(b_rhs) # Use RHS as initial guess or zeros
                    result_vec, info = spla.gmres(H_solve, b_rhs, x0=x0_init, tol=1e-8, maxiter=50)
                    if info != 0: raise np.linalg.LinAlgError(f"GMRES did not converge cleanly (info={info}).")

                else:
                    raise ValueError(f"Unknown solver method: {current_method_for_try}")

                if not np.all(np.isfinite(result_vec)): # Critical check for NaN/Inf in result
                    raise ValueError("Solution vector not finite after solve.")

                return result_vec, num_psi_attempts # Successful solve and number of attempts

            except (np.linalg.LinAlgError, ValueError, TypeError) as e:
                # If solve fails: first time, switch to fallback method. Subsequent times, retry with stronger Psi.
                if current_method_for_try == self.preferred_method and self.preferred_method != self.fallback_method and num_psi_attempts == 0:
                    current_method_for_try = self.fallback_method # Switch once to fallback on first failure
                    num_psi_attempts = 0 # Reset PSI attempts for the newly chosen solver
                    continue

                num_psi_attempts += 1

        # If all attempts exhausted without success
        raise RuntimeError(f"InverseIterateSolver failed all {self.max_attempts} attempts for {self.preferred_method} and {self.fallback_method}.")


# --- Solution Candidate Class (Represents a single hypothesis/solution candidate) ---
# Each candidate is an autonomous agent making local progress based on MAUS's global strategy.
class SolutionCandidate:
    _candidate_id_counter = 0
    # Internal states define candidate behavior
    class State(Enum):
        EXPLORING = 1  # In search phase, might take larger/randomized steps
        REFINING = 2   # Has found a promising region, focusing on tighter convergence
        STUCK = 3      # Repeatedly failed or diverged locally. Needs intervention or retirement.
        CONVERGED = 4  # Has met convergence criteria
        RETIRED = 5    # Has been pruned from population due to redundancy or persistent failure

    def __init__(self, problem_matrix, problem_type, N_diag, initial_lambda=None, initial_v=None, initial_x=None, initial_u=None, initial_sigma=None, initial_weight=0.01):
        self.id = SolutionCandidate._candidate_id_counter
        SolutionCandidate._candidate_id_counter += 1

        self.N_diag = N_diag # Dimension for square operations (e.g., N for Eigen)
        self.M_rows, self.M_cols = problem_matrix.shape # Actual dimensions of input matrix
        self.problem_type = problem_type
        self.problem_matrix = problem_matrix
        self.b_vector = None

        # Solution parameters (type-specific containers)
        self.lambda_k = initial_lambda
        self.v_k = initial_v
        self.x_k = initial_x
        self.sigma_k = initial_sigma
        self.u_k = initial_u
        self.right_v_k = initial_v

        # Candidate State and confidence tracking
        self.state = SolutionCandidate.State.EXPLORING # Initial state
        self.w_k = initial_weight # Confidence/weight
        self.residual_k = float('inf') # Current residual (lower is better)
        self.prev_residual = float('inf') # Residual from previous step (for adaptation)
        self.alpha_local_step = GLOBAL_DEFAULT_ALPHA_V_INITIAL # Individual step size for reactive gradient

        self.stuck_counter = 0 # Counts how many consecutive times the candidate needed brute-force intervention
        self.local_psi_retries_needed = 0 # Records retries needed by InverseIterateSolver for last update
        self.num_resets = 0 # Counts total times its internal state was randomly re-initialized due to failures

        # History (for debugging and learning over time)
        self.param_history = []
        self.residual_history = []

        self.initialize_random_solution() # Set initial state of solution parameters


    def initialize_random_solution(self):
        # Helper to create a random normalized complex vector
        rand_vec_init = lambda N: (np.random.rand(N) + 1j * np.random.rand(N)).astype(np.complex128)
        # Use simple perturbation or full random based on a threshold on stuck_counter if applicable.
        # For MAUS's internal logic, this just means "reinitialize from scratch if I fail this".
        norm_rand_vec = lambda v_raw: v_raw / np.linalg.norm(v_raw) if np.linalg.norm(v_raw) > 1e-10 else rand_vec_init(v_raw.shape[0]) # Defensive normalization


        if self.problem_type == ProblemType.EIGENVALUE:
            self.v_k = norm_rand_vec(rand_vec_init(self.N_diag))
            self.lambda_k = (random.random() * 5 - 2.5 + 1j * (random.random() * 5 - 2.5)) # Random complex lambda

        elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM:
            self.x_k = norm_rand_vec(rand_vec_init(self.N_diag)) * random.uniform(0.1, 10.0) # Random magnitude initial solution

        elif self.problem_type == ProblemType.SVD:
            self.u_k = norm_rand_vec(rand_vec_init(self.M_rows))
            self.right_v_k = norm_rand_vec(rand_vec_init(self.M_cols))
            self.sigma_k = 1.0

        # Store initial (possibly inf) residual and solution for history.
        self.param_history.append(self.get_current_solution_params())
        self.residual_history.append(self.residual_k)

    def update_solution_step(self, current_matrix_A, b_vector=None, strat_params=None, global_knowledge=None):
        self.b_vector = b_vector # `b` vector for Ax=b problems
        self.prev_residual = self.residual_k # Store last residual for step-size adaptation

        # Parameters for local solver instance from global strategy & knowledge
        overall_psi_aggression_factor = strat_params.get('overall_psi_aggression_factor', 1.0)
        max_psi_retries_global = strat_params.get('max_psi_retries', GLOBAL_MAX_PSI_ATTEMPTS)
        local_solver_preference = global_knowledge.get('local_solver_preference', 'direct_solve') # 'direct_solve' or 'iterative_gmres'
        is_matrix_sparse = global_knowledge.get('is_sparse_problem', False)

        solver_instance = InverseIterateSolver(self.N_diag, GLOBAL_DEFAULT_PSI_EPSILON_BASE * overall_psi_aggression_factor,
                                                max_psi_retries_global, local_solver_preference, is_matrix_sparse)

        # Branch based on problem type for specific update logic
        if self.problem_type == ProblemType.SVD:
            try:
                # SVD works via alternating matrix-vector products (like power method variants).
                # If a vector's norm is tiny, we might add noise or reinitialize.
                if np.linalg.norm(self.right_v_k) < 1e-10:
                    self.right_v_k = (np.random.rand(self.M_cols) + 1j * np.random.rand(self.M_cols)); self.right_v_k /= np.linalg.norm(self.right_v_k)
                    self.stuck_counter += 1; self.num_resets += 1;
                    raise ValueError("SVD right_v_k collapsed. Reinitializing.")

                temp_u_k = current_matrix_A @ self.right_v_k
                self.sigma_k = np.linalg.norm(temp_u_k) # Best singular value estimate
                self.u_k = temp_u_k / (self.sigma_k if self.sigma_k > 1e-10 else 1.0) # Normalize `u`

                if np.linalg.norm(self.u_k) < 1e-10: # Check if u also collapsed (potential error propagation)
                     self.u_k = (np.random.rand(self.M_rows)+1j*np.random.rand(self.M_rows)); self.u_k /= np.linalg.norm(self.u_k)
                     self.stuck_counter += 1; self.num_resets += 1;
                     raise ValueError("SVD u_k collapsed. Reinitializing.")

                temp_v_k = current_matrix_A.conj().T @ self.u_k
                self.sigma_k = max(self.sigma_k, np.linalg.norm(temp_v_k)) # Take the maximum sigma from both updates
                self.right_v_k = temp_v_k / (np.linalg.norm(temp_v_k) if np.linalg.norm(temp_v_k) > 1e-10 else 1.0)

                # Small sigma might indicate convergence to zero singular value, not necessarily a failure.
                if self.sigma_k < GLOBAL_SIGMA_SIMILARITY_TOL_ABS / 100 : # If sigma is tiny even lower than default (often implies it's really 0)
                    self.residual_k = strat_params.get('current_convergence_threshold', 1e-6) * 0.1 # Set very small residual to acknowledge "convergence to zero sigma"
                    self.state = SolutionCandidate.State.CONVERGED # It found a very small sigma and solved for it
                    self.stuck_counter = 0 # No longer stuck
                    # Ensure u and v are well-defined for downstream usage if sigma is zero
                    if np.linalg.norm(self.u_k) < 1e-10: self.u_k = np.ones(self.M_rows, dtype=np.complex128)/np.sqrt(self.M_rows)
                    if np.linalg.norm(self.right_v_k) < 1e-10: self.right_v_k = np.ones(self.M_cols, dtype=np.complex128)/np.sqrt(self.M_cols)

                else: # Otherwise, standard processing
                    self.stuck_counter = max(0, self.stuck_counter - 1) # Reduce stuck counter on successful SVD step

            except (RuntimeError, ValueError, np.linalg.LinAlgError) as e: # Catch any errors from SVD path or its internal vector normalization
                self.stuck_counter += 1; self.w_k *= 0.001; self.alpha_local_step *= 0.5
                self.state = SolutionCandidate.State.STUCK
                if self.stuck_counter >= GLOBAL_MAX_STUCK_FOR_RETIREMENT: self.state = SolutionCandidate.State.RETIRED
                # If SVD method explicitly threw error, re-randomize for brute-force exploration
                self.u_k = (np.random.rand(self.M_rows)+1j*np.random.rand(self.M_rows))/np.sqrt(self.M_rows)
                self.right_v_k = (np.random.rand(self.M_cols)+1j*np.random.rand(self.M_cols))/np.sqrt(self.M_cols)
                self.sigma_k = 1.0

        # --- Common update block for Eigenvalue and SolveLinearSystem problems (using InverseIterateSolver) ---
        else:
            target_A_for_solve = current_matrix_A
            rhs_for_solve = None
            current_main_vec_ref = None

            if self.problem_type == ProblemType.EIGENVALUE:
                if np.linalg.norm(self.v_k) < 1e-10:
                    self.v_k = (np.random.rand(self.N_diag) + 1j*np.random.rand(self.N_diag)); self.v_k /= np.linalg.norm(self.v_k)
                    self.stuck_counter += 1; self.num_resets += 1;
                    raise ValueError("Eigenvector collapsed. Reinitializing random vector for a fresh start.") # This restarts `try` block, potentially with a new `Psi`
                self.lambda_k = (self.v_k.conj().T @ current_matrix_A @ self.v_k) / (self.v_k.conj().T @ self.v_k) # Reactive lambda update
                target_A_for_solve = current_matrix_A - self.lambda_k * np.eye(self.N_diag, dtype=current_matrix_A.dtype)
                rhs_for_solve = self.v_k # `v_k` serves as the right-hand side for (A-λI)z = v
                current_main_vec_ref = self.v_k

            elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM:
                target_A_for_solve = current_matrix_A
                rhs_for_solve = self.b_vector
                current_main_vec_ref = self.x_k

            try:
                new_vec_raw, self.local_psi_retries_needed = solver_instance.solve(target_A_for_solve, rhs_for_solve, self.stuck_counter)

                # Apply alpha_local_step for controlled blend/step. This prevents overshooting and aids stability.
                if self.problem_type == ProblemType.EIGENVALUE:
                    # Blends the old `v_k` with the `new_vec_raw` in the direction of the solution
                    self.v_k = (1.0 - self.alpha_local_step) * self.v_k + self.alpha_local_step * new_vec_raw
                    self.v_k /= np.linalg.norm(self.v_k) if np.linalg.norm(self.v_k) > 1e-10 else (np.random.rand(self.N_diag)+1j*np.random.rand(self.N_diag))/np.sqrt(self.N_diag) # Normalize and protect against 0-norm
                elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM:
                    # For Ax=b, `new_vec_raw` IS the candidate `X`. Alpha blends current `x_k` with this newly calculated `X`.
                    self.x_k = (1.0 - self.alpha_local_step) * current_main_vec_ref + self.alpha_local_step * new_vec_raw

                self.stuck_counter = max(0, self.stuck_counter - 1) # Success means reduction in stuckness

            except (RuntimeError, ValueError) as e: # Catch InverseIterateSolver failure (ran out of PSI/solver types)
                self.stuck_counter += 1
                self.w_k *= 0.001 # Penalize candidate weight
                self.alpha_local_step = max(self.alpha_local_step * 0.5, 1e-6) # Aggressive step size reduction
                if self.stuck_counter >= GLOBAL_MAX_STUCK_FOR_RETIREMENT: # Candidate failed too many times, retire it
                    self.state = SolutionCandidate.State.RETIRED
                    self.num_resets += 1 # Count how many were completely reset and retired
                else: # Otherwise, mark as stuck for now and retry with random state next
                    self.state = SolutionCandidate.State.STUCK
                    self.initialize_random_solution() # Reset vector/params, retaining `stuck_counter`


        # --- Common Residual Calculation & History Logging (Regardless of previous branch) ---
        A = self.problem_matrix # Get the current (potentially updated, e.g., dynamic A(t)) matrix for residual calc
        if self.problem_type == ProblemType.EIGENVALUE:
            self.residual_k = np.linalg.norm(A @ self.v_k - self.lambda_k * self.v_k)
        elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM:
            self.residual_k = np.linalg.norm(A @ self.x_k - self.b_vector)
        # SVD residual is calculated in its update path; just verify it's not infinite now
        elif self.problem_type == ProblemType.SVD:
            self.residual_k = np.linalg.norm(A @ self.right_v_k - self.sigma_k * self.u_k) + \
                              np.linalg.norm(A.conj().T @ self.u_k - self.sigma_k * self.right_v_k)

        # Append to history, now guaranteed to have all parameters from whichever branch
        self.param_history.append(self.get_current_solution_params())
        self.residual_history.append(self.residual_k)

        # Adaptive alpha_local_step and candidate State Transition (Common)
        if self.prev_residual > 1e-10:
            if self.residual_k < self.prev_residual * 0.9: # Significant improvement (reward)
                self.alpha_local_step = min(self.alpha_local_step * 1.1, 1.0)
                if self.state != SolutionCandidate.State.CONVERGED: self.state = SolutionCandidate.State.REFINING
            elif self.residual_k > self.prev_residual * 1.5 and self.prev_residual > GLOBAL_CONVERGENCE_RESIDUAL_TOL * 10: # Diverging significantly, and not already very close to converged
                self.alpha_local_step = max(self.alpha_local_step * 0.5, 1e-6) # Aggressively dampen step size
                if self.state != SolutionCandidate.State.CONVERGED: self.state = SolutionCandidate.State.STUCK # Mark as stuck for current round, allow strategies to handle
            else: # Stagnant or minor progress (decay learning rate, and if it wasn't already in another state)
                self.alpha_local_step = max(self.alpha_local_step * 0.95, 1e-6) # Gradually decrease exploration size
                if self.state not in [SolutionCandidate.State.CONVERGED, SolutionCandidate.State.STUCK, SolutionCandidate.State.RETIRED]:
                     self.state = SolutionCandidate.State.EXPLORING # Continue searching for better paths

        # Final Convergence check (can switch state to CONVERGED)
        if self.residual_k < strat_params.get('current_convergence_threshold', GLOBAL_CONVERGENCE_RESIDUAL_TOL) and \
           np.all(np.isfinite(self.get_current_solution_params()[-1])): # Final check for numerical stability of result
            self.state = SolutionCandidate.State.CONVERGED
            self.w_k = 1.0 # Max confidence for converged solutions
            self.stuck_counter = 0 # Reset stuck counter
            self.alpha_local_step = 0.0 # Halt stepping for converged solutions

    def get_current_solution_params(self):
        # Returns the relevant solution parameters as a tuple
        if self.problem_type == ProblemType.EIGENVALUE: return (self.lambda_k, self.v_k)
        elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM: return (self.x_k,)
        elif self.problem_type == ProblemType.SVD: return (self.sigma_k, self.u_k, self.right_v_k)
        return None

    # Segment 5: Particle Simulation Functions (Auxiliary & Analysis)

# Function to Update Bonds (remains the same, uses particle_parameters)
def update_bonds(positions, orientations, types, bonds, particle_parameters, current_step, all_patch_data):
    """
    Updates the list of active bonds based on formation and breaking criteria.
    Supports both center-to-center and patch-based bonding.
    Uses particle_parameters for bonding criteria and patch definitions.

    Args:
        positions (np.ndarray): Current positions of particles.
        orientations (np.ndarray): Current orientations of particles.
        types (np.ndarray): Types of particles.
        bonds (dict): Current dictionary of active bonds.
        particle_parameters (dict): Dictionary of particle simulation parameters.
        current_step (int): The current simulation step number.
        all_patch_data (list of lists): Data structure containing info for all patches.

    Returns:
        dict: Updated dictionary of active bonds.
    """
    bonding_params = particle_parameters["bonding"]
    if not bonding_params["enabled"]:
        return bonds # Bonding is disabled

    formation_criteria = bonding_params["formation_criteria"]
    distance_tolerance = formation_criteria["distance_tolerance"]
    # Ensure patch_type_compatibility_matrix is a numpy array of bools
    patch_type_compatibility_matrix = np.array(formation_criteria.get("patch_type_compatibility_matrix", [[]]), dtype=bool)
    orientation_alignment_tolerance = formation_criteria["orientation_alignment_tolerance"]
    bond_break_distance = bonding_params["bond_break_distance"]
    # bond_types_param = tuple(bonding_params["bond_types"]) # Not directly used here, types in patches used

    patch_params = particle_parameters["forces"].get("patch_params", {})
    patch_enabled = patch_params.get("enabled", False)
    patch_definitions = patch_params.get("patch_definitions", {})

    updated_bonds = bonds.copy()

    num_particles = positions.shape[0]
    if num_particles == 0 or not all_patch_data : return updated_bonds # No particles or patch data means no bonds

    boundaries = particle_parameters["boundaries"]
    box_size = np.array([boundaries["x_max"] - boundaries["x_min"], boundaries["y_max"] - boundaries["y_min"]])

    # --- Bond Breaking ---
    bonds_to_break = []
    for bond_key in updated_bonds.keys():
         is_patch_bond = isinstance(bond_key, tuple) and len(bond_key) == 2 and isinstance(bond_key[0], tuple)
         if is_patch_bond:
              (i, p_idx_i), (j, p_idx_j) = bond_key
              # Check particles/patches exist and indices valid
              if not (i < num_particles and j < num_particles and i < len(all_patch_data) and j < len(all_patch_data) and \
                  p_idx_i < len(all_patch_data[i]) and p_idx_j < len(all_patch_data[j])):
                   bonds_to_break.append(bond_key); continue

              patch_i_pos = all_patch_data[i][p_idx_i]["position"]
              patch_j_pos = all_patch_data[j][p_idx_j]["position"]
              r_patch_raw = patch_i_pos - patch_j_pos
              r_patch_pbc = r_patch_raw - box_size * np.round(r_patch_raw / box_size)
              if np.linalg.norm(r_patch_pbc) > bond_break_distance:
                   bonds_to_break.append(bond_key)
         # else: # Center-center bonds are not formed by default in this configuration, but can be added
             # i,j = bond_key
             # if not (i < num_particles and j < num_particles): bonds_to_break.append(bond_key); continue
             # r_cc_raw = positions[j,:] - positions[i,:]
             # r_cc_pbc = r_cc_raw - box_size * np.round(r_cc_raw / box_size)
             # if np.linalg.norm(r_cc_pbc) > bonding_params.get("bond_break_distance", 2.0): # Use specific param for CC if needed
             #      bonds_to_break.append(bond_key)

    for bond_key in bonds_to_break:
        if bond_key in updated_bonds: del updated_bonds[bond_key]


    # --- Bond Formation (Patch-Based) ---
    if patch_enabled and patch_definitions:
        # Determine search cutoff based on potential range or break distance
        formation_search_cutoff = bond_break_distance # Max distance to consider formation
        max_patch_extension = 0
        for p_type, p_defs in patch_definitions.items():
            for p_spec in p_defs: max_patch_extension = max(max_patch_extension, p_spec.get("distance",0))


        for i in range(num_particles):
            for j in range(i + 1, num_particles):
                if not all_patch_data[i] or not all_patch_data[j]: continue

                r_ij_center_vec = positions[i, :] - positions[j, :]
                r_ij_center_vec_pbc = r_ij_center_vec - box_size * np.round(r_ij_center_vec / box_size)
                if np.linalg.norm(r_ij_center_vec_pbc) > formation_search_cutoff + 2 * max_patch_extension: continue


                for p_idx_i, patch_i_data in enumerate(all_patch_data[i]):
                    for p_idx_j, patch_j_data in enumerate(all_patch_data[j]):
                        potential_bond_key = tuple(sorted(((i, p_idx_i), (j, p_idx_j))))
                        if potential_bond_key in updated_bonds: continue

                        patch_i_pos, patch_j_pos = patch_i_data["position"], patch_j_data["position"]
                        patch_i_type, patch_j_type = patch_i_data["patch_type"], patch_j_data["patch_type"]

                        # Type compatibility check
                        if not (0 <= patch_i_type < patch_type_compatibility_matrix.shape[0] and \
                                0 <= patch_j_type < patch_type_compatibility_matrix.shape[1] and \
                                patch_type_compatibility_matrix[patch_i_type, patch_j_type]):
                             continue

                        r_patch_raw = patch_i_pos - patch_j_pos
                        r_patch_pbc = r_patch_raw - box_size * np.round(r_patch_raw / box_size)
                        r_patch_mag = np.linalg.norm(r_patch_pbc)

                        # Distance criteria for formation
                        # Use characteristic distance from bond potential type
                        char_bond_dist = bonding_params.get("patch_bond_potential", {}).get("harmonic", {}).get("bond_distance", bond_break_distance)
                        if bonding_params.get("patch_bond_potential",{}).get("type") == "lennard_jones":
                            char_bond_dist = bonding_params.get("patch_bond_potential",{}).get("lennard_jones",{}).get("sigma", bond_break_distance)
                        elif bonding_params.get("patch_bond_potential",{}).get("type") == "square_well":
                            char_bond_dist = bonding_params.get("patch_bond_potential",{}).get("square_well",{}).get("sigma", bond_break_distance)


                        form_dist_met = False
                        if r_patch_mag < bond_break_distance: # Must be within break distance
                             if abs(r_patch_mag - char_bond_dist) < distance_tolerance : form_dist_met = True
                             if bonding_params.get("patch_bond_potential",{}).get("type") == "square_well":
                                 sw_sig = bonding_params.get("patch_bond_potential",{}).get("square_well",{}).get("sigma", bond_break_distance)
                                 sw_lam = bonding_params.get("patch_bond_potential",{}).get("square_well",{}).get("lambda", 1.0)
                                 if sw_sig <= r_patch_mag < sw_sig * sw_lam : form_dist_met = True


                        if form_dist_met:
                            # Orientation alignment (if enabled)
                            orient_align_met = True
                            if orientation_alignment_tolerance is not None and orientation_alignment_tolerance < np.pi: # Check if non-trivial tolerance
                                ideal_angle_map = particle_parameters["forces"]["orientation_potential"]["bond_angle_potential"].get("ideal_angle_mapping", {})

                                # Patch i orientation
                                particle_type_i_int = types[i]
                                # Defensive initialization for pdefs_i
                                pdefs_i = []
                                if particle_type_i_int in patch_definitions:
                                     pdefs_i = patch_definitions.get(particle_type_i_int)
                                elif str(particle_type_i_int) in patch_definitions: # Fallback to string key if int key not found
                                     pdefs_i = patch_definitions.get(str(particle_type_i_int))
                                # Ensure it is a list
                                if not isinstance(pdefs_i, list): pdefs_i = []

                                if p_idx_i >= len(pdefs_i): continue
                                angle_rel_i = pdefs_i[p_idx_i].get("angle_relative_to_particle",0.0)
                                abs_angle_i = orientations[i] + angle_rel_i
                                patch_i_orient_vec = np.array([np.cos(abs_angle_i), np.sin(abs_angle_i)])
                                ideal_i = ideal_angle_map.get(patch_i_type, 0.0) # ideal_angle_map has int keys
                                angle_i_to_bond = np.arctan2(patch_i_orient_vec[0]*(r_patch_pbc[1]/r_patch_mag) - patch_i_orient_vec[1]*(r_patch_pbc[0]/r_patch_mag), np.dot(patch_i_orient_vec, r_patch_pbc/r_patch_mag))
                                dev_i = abs(np.mod(angle_i_to_bond - ideal_i + np.pi, 2*np.pi) - np.pi)


                                # Patch j orientation
                                particle_type_j_int = types[j]
                                # Defensive initialization for pdefs_j
                                pdefs_j = []
                                if particle_type_j_int in patch_definitions:
                                     pdefs_j = patch_definitions.get(particle_type_j_int)
                                elif str(particle_type_j_int) in patch_definitions: # Fallback to string key
                                     pdefs_j = patch_definitions.get(str(particle_type_j_int))
                                # Ensure it is a list
                                if not isinstance(pdefs_j, list): pdefs_j = []

                                if p_idx_j >= len(pdefs_j): continue
                                angle_rel_j = pdefs_j[p_idx_j].get("angle_relative_to_particle",0.0)
                                abs_angle_j = orientations[j] + angle_rel_j
                                patch_j_orient_vec = np.array([np.cos(abs_angle_j), np.sin(abs_angle_j)])
                                ideal_j = ideal_angle_map.get(patch_j_type, 0.0)
                                angle_j_to_bond = np.arctan2(patch_j_orient_vec[0]*(-bond_vec_dir[1])-patch_j_orient_vec[1]*(-bond_vec_dir[0]), np.dot(patch_j_orient_vec,-bond_vec_dir))
                                dev_j = abs(np.mod(angle_j_to_bond - ideal_j + np.pi, 2*np.pi) - np.pi)


                                if dev_i > orientation_alignment_tolerance or dev_j > orientation_alignment_tolerance:
                                     orient_align_met = False

                            if orient_align_met:
                                 # Form bond
                                 initial_strength = bonding_params.get("patch_bond_potential",{}).get("harmonic",{}).get("bond_strength", 100.0) # Store initial harmonic for adaptive
                                 updated_bonds[potential_bond_key] = {'formed_step': current_step, 'initial_strength': initial_strength, 'patch_pair': (p_idx_i, p_idx_j)}

    return updated_bonds


# Function for Particle Creation (remains the same, uses particle_parameters)
def create_particles(positions, velocities, orientations, types, masses, bonds, particle_parameters, current_step, all_patch_data):
    """
    Handles particle creation based on defined criteria.
    (Placeholder - actual creation logic needs to be implemented based on triggers,
    possibly influenced by brain state).
    Uses particle_parameters for creation criteria and new particle properties.
    """
    creation_params = particle_parameters.get("particle_creation", {})
    if not creation_params.get("enabled", False) or random.random() > creation_params.get("creation_rate",0.0): # Stochastic rate
        return positions, velocities, orientations, types, masses, bonds, all_patch_data, len(positions)

    num_existing_particles = len(positions)
    new_particle_config = creation_params.get("new_particle", {})
    trigger_config = creation_params.get("trigger", {})
    # Placeholder for trigger logic (e.g. a specific particle type initiating creation)
    # For simplicity, let's try to add one particle if any particle exists
    if num_existing_particles == 0:
        return positions, velocities, orientations, types, masses, bonds, all_patch_data, num_existing_particles


    # Select a random existing particle as parent (very simplified trigger)
    parent_idx = random.randrange(num_existing_particles)
    parent_pos = positions[parent_idx]
    parent_vel = velocities[parent_idx]
    parent_orient = orientations[parent_idx]
    # parent_ang_vel = angular_velocities[parent_idx] # If angular_velocities available


    new_type = new_particle_config.get("type", 0)
    new_pos_offset = np.array([ (np.random.rand()-0.5)*2 , (np.random.rand()-0.5)*2 ]) # Random offset
    new_pos = parent_pos + new_pos_offset
    new_vel = parent_vel * new_particle_config.get("initial_velocity_scale", 0.1)

    new_orient_config = new_particle_config.get("angular_initialization", particle_parameters["initial_conditions"]["new_particle_angular_initialization"])
    new_orient = parent_orient if new_orient_config["orientation_type"] == "copy_parent" else new_orient_config["orientation_angle"]
    # new_ang_vel = parent_ang_vel * new_orient_config["angular_velocity_scale"] if new_orient_config["angular_velocity_type"] == "copy_scaled_parent" else 0.0


    new_mass = particle_parameters["initial_conditions"]["mass_mapping"].get(new_type, 1.0)

    # Append new particle
    positions = np.vstack([positions, new_pos])
    velocities = np.vstack([velocities, new_vel])
    orientations = np.append(orientations, new_orient)
    types = np.append(types, new_type)
    masses = np.append(masses, new_mass)
    # angular_velocities = np.append(angular_velocities, new_ang_vel) # If tracking
    # accelerations & angular_accelerations will need to be resized too

    # Update all_patch_data: add empty list for the new particle
    all_patch_data.append([]) # This will be properly populated in next force calculation

    num_particles = len(positions)
    # print(f"Particle created. Total particles: {num_particles}")
    return positions, velocities, orientations, types, masses, bonds, all_patch_data, num_particles


# Function for Particle Deletion (remains the same, uses particle_parameters)
def delete_particles(positions, velocities, orientations, types, masses, bonds, particle_parameters, current_step, all_patch_data):
    """
    Handles particle deletion based on defined criteria.
    (Placeholder - actual deletion logic needs to be implemented based on triggers,
    possibly influenced by brain state).
    Uses particle_parameters for deletion criteria.
    """
    deletion_params = particle_parameters.get("particle_deletion", {})
    if not deletion_params.get("enabled", False) or random.random() > deletion_params.get("deletion_rate",0.0):
        return positions, velocities, orientations, types, masses, bonds, all_patch_data, len(positions)

    num_particles_before_del = len(positions)
    if num_particles_before_del == 0: return positions, velocities, orientations, types, masses, bonds, all_patch_data, 0


    trigger = deletion_params.get("trigger", {})
    condition = trigger.get("condition", None)
    buffer_distance = trigger.get("buffer_distance", 0.0)
    particles_to_delete_indices = []

    if condition == "out_of_bounds":
         boundaries = particle_parameters["boundaries"]
         x_min, x_max = boundaries["x_min"], boundaries["x_max"]
         y_min, y_max = boundaries["y_min"], boundaries["y_max"]
         for i in range(num_particles_before_del):
              if (positions[i, 0] < x_min - buffer_distance or positions[i, 0] > x_max + buffer_distance or
                  positions[i, 1] < y_min - buffer_distance or positions[i, 1] > y_max + buffer_distance):
                   particles_to_delete_indices.append(i)
    # Add other deletion conditions if needed

    if particles_to_delete_indices:
         keep_mask = np.ones(num_particles_before_del, dtype=bool)
         keep_mask[particles_to_delete_indices] = False

         positions = positions[keep_mask]
         velocities = velocities[keep_mask]
         orientations = orientations[keep_mask]
         types = types[keep_mask]
         masses = masses[keep_mask]
         # Resize accelerations if they are managed outside:
         # accelerations = accelerations[keep_mask]
         # angular_velocities = angular_velocities[keep_mask]
         # angular_accelerations = angular_accelerations[keep_mask]

         all_patch_data_updated = [patch_list for i, patch_list in enumerate(all_patch_data) if keep_mask[i]]
         all_patch_data = all_patch_data_updated

         # Update bonds: remove bonds involving deleted particles and re-index remaining
         old_to_new_map = {old_idx: new_idx for new_idx, old_idx in enumerate(np.where(keep_mask)[0])}
         updated_bonds = {}
         for bond_key, bond_info in bonds.items():
             is_patch_bond = isinstance(bond_key, tuple) and len(bond_key) == 2 and isinstance(bond_key[0], tuple)
             if is_patch_bond:
                  (i, p_idx_i), (j, p_idx_j) = bond_key
                  if i in old_to_new_map and j in old_to_new_map:
                       updated_bonds[tuple(sorted(((old_to_new_map[i], p_idx_i), (old_to_new_map[j], p_idx_j))))] = bond_info
             # else: # Center-center bond
                 # i,j = bond_key
                 # if i in old_to_new_map and j in old_to_new_map:
                 #      updated_bonds[tuple(sorted((old_to_new_map[i], old_to_new_map[j])))] = bond_info
         bonds = updated_bonds
         # print(f"Deleted {len(particles_to_delete_indices)} particles. Remaining: {len(positions)}")


    return positions, velocities, orientations, types, masses, bonds, all_patch_data, len(positions)


# Function for State Change on Bond Formation (remains the same, uses particle_parameters)
def apply_state_change_on_bond_form(types, bonds, particle_parameters, current_step):
    """
    Changes the type of a particle when a specific bond is formed.
    (Placeholder - actual logic needs to be implemented based on triggers).
    Uses particle_parameters for state change criteria.
    """
    state_change_params = particle_parameters.get("state_change", {})
    if not state_change_params.get("enabled", False):
        return types

    on_bond_form_params = state_change_params.get("on_bond_form", {})
    from_type_config = on_bond_form_params.get("from_type", None) # This could be an int
    to_type_config = on_bond_form_params.get("to_type", None) # This could be an int

    if from_type_config is None or to_type_config is None:
         return types

    updated_types = np.copy(types)
    num_particles = updated_types.shape[0]

    for bond_key, bond_info in bonds.items():
         if bond_info.get('formed_step', -1) == current_step: # Bond formed in current step
              is_patch_bond = isinstance(bond_key, tuple) and len(bond_key) == 2 and isinstance(bond_key[0], tuple)
              if is_patch_bond:
                   (i, _), (j, _) = bond_key # Don't need patch indices for type change of particle
                   if i < num_particles and updated_types[i] == from_type_config: updated_types[i] = to_type_config
                   if j < num_particles and updated_types[j] == from_type_config: updated_types[j] = to_type_config
              # else: # Center-center bond
                   # i,j = bond_key
                   # if i < num_particles and updated_types[i] == from_type_config : updated_types[i] = to_type_config
                   # if j < num_particles and updated_types[j] == from_type_config : updated_types[j] = to_type_config
    return updated_types


# Function to Get Cluster Labels (remains the same)
def get_cluster_labels_for_frame(positions, bonds, num_particles):
    """
    Identifies clusters of particles based on active bonds.
    """
    if num_particles == 0:
        return np.array([], dtype=int)

    graph = scipy.sparse.lil_matrix((num_particles, num_particles), dtype=int)
    for bond_key in bonds.keys():
         is_patch_bond = isinstance(bond_key, tuple) and len(bond_key) == 2 and isinstance(bond_key[0], tuple)
         if is_patch_bond:
              (i, _), (j, _) = bond_key # Particle indices are i and j
              if i < num_particles and j < num_particles:
                   graph[i, j] = 1; graph[j, i] = 1
         # else: # Center-center bond
             # i, j = bond_key
             # if i < num_particles and j < num_particles:
             #      graph[i, j] = 1; graph[j, i] = 1

    n_components, labels = scipy.sparse.csgraph.connected_components(graph, directed=False, connection='weak') # Weak for undirected
    return labels


# Function to Save Simulation State (remains the same, uses particle_parameters)
def save_simulation_state(step, positions, velocities, orientations, types, masses, bonds, particle_parameters):
    """
    Saves the current state of the particle simulation to files.
    Uses particle_parameters for the save directory.
    """
    save_directory = particle_parameters["saving"]["save_directory"]
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Convert bonds to JSON-serializable format (list of lists for keys, info as dict)
    serializable_bonds = {}
    for k, v in bonds.items():
        # Key: Convert ((p1_idx, patch1_idx), (p2_idx, patch2_idx)) to string "p1_patch1-p2_patch2"
        # Or handle simpler int-tuple keys for center-center bonds.
        # For pickle, tuples are fine as keys. If this was for JSON, more care needed.
        # Since it's pickle, we can store bonds dictionary directly.
        serializable_bonds[k] = v


    state = {
        "step": step,
        "positions": positions.tolist(),
        "velocities": velocities.tolist(),
        "orientations": orientations.tolist(),
        "types": types.tolist(),
        "masses": masses.tolist(),
        "bonds": serializable_bonds, # Store the dict as-is for pickle
        "parameters": particle_parameters
    }

    filepath = os.path.join(save_directory, f"particle_state_step_{step:06d}.pkl")
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    except Exception as e:
        print(f"Error saving particle simulation state to {filepath}: {e}")


# Function to Load Simulation State (remains the same, uses particle_parameters)
def load_simulation_state(particle_parameters_input): # Rename to avoid conflict with global
    """
    Loads the latest particle simulation state from files in the load directory.
    Uses particle_parameters_input for the load directory.
    """
    load_directory = particle_parameters_input["saving"]["load_directory"]
    if not os.path.exists(load_directory):
        print(f"Particle load directory '{load_directory}' not found. Starting from initial conditions.")
        return None

    state_files = [f for f in os.listdir(load_directory) if f.startswith("particle_state_step_") and f.endswith(".pkl")]
    if not state_files:
        print(f"No particle simulation state files found in '{load_directory}'. Starting from initial conditions.")
        return None

    try: # Add try block for parsing step numbers
        latest_step = max([int(f.split('_')[-1].split('.')[0]) for f in state_files])
    except ValueError:
        print(f"Error parsing step numbers from filenames in '{load_directory}'. Starting fresh.")
        return None

    latest_filepath = os.path.join(load_directory, f"particle_state_step_{latest_step:06d}.pkl")

    try:
        with open(latest_filepath, 'rb') as f:
            state = pickle.load(f)
        print(f"Particle simulation state loaded successfully from '{latest_filepath}'")

        # Use parameters from the loaded state primarily
        loaded_parameters = state.get("parameters", particle_parameters_input)
        # Override with current script's visualization params if desired, or merge carefully
        # For now, strictly use loaded_parameters for consistency of the loaded state.

        loaded_types_np = np.array(state["types"], dtype=int)
        vis_params = loaded_parameters.get("visualization", {})
        patches_vis_params = vis_params.get("patches", {})
        color_mapping_from_loaded = patches_vis_params.get("color_mapping", {})
        # Ensure color_mapping keys are int for lookup, as types are int
        color_mapping_int_keys = {int(k):v for k,v in color_mapping_from_loaded.items()} if isinstance(color_mapping_from_loaded,dict) else {}


        loaded_colors = [color_mapping_int_keys.get(t, 'gray') for t in loaded_types_np]


        # Accelerations and angular velocities/accelerations are re-calculated at simulation start.
        num_loaded_particles = len(state["positions"])
        return (
            state["step"],
            np.array(state["positions"]),
            np.array(state["velocities"]),
            np.zeros((num_loaded_particles,2)) if num_loaded_particles > 0 else np.array([]).reshape(0,2), # Accelerations placeholder
            state.get("bonds", {}), # Bonds directly from pickle
            loaded_types_np,
            np.array(state["masses"]),
            loaded_colors,
            np.array(state["orientations"]),
            np.zeros(num_loaded_particles) if num_loaded_particles > 0 else np.array([]), # Angular velocities placeholder
            np.zeros(num_loaded_particles) if num_loaded_particles > 0 else np.array([]), # Angular accelerations placeholder
            num_loaded_particles,
            loaded_parameters, # Return the parameters that were part of the loaded state
            [] # Placeholder for all_patch_data, regenerated in first step
        )

    except FileNotFoundError:
        print(f"Error loading particle state from '{latest_filepath}'. File not found.")
        return None
    except Exception as e:
        print(f"An error occurred loading particle simulation state: {e}. Starting from initial conditions.")
        return None


# Analysis Function: Radial Distribution Function (RDF) (remains the same, uses particle_parameters)
def calculate_radial_distribution_function(positions_history, particle_parameters):
    """
    Calculates the radial distribution function (g(r)) for the particle system.
    Calculates g(r) averaged over specified frames.
    Uses particle_parameters for analysis settings and boundaries.
    Requires positions_history.
    """
    analysis_params = particle_parameters["analysis"]
    rdf_dr = analysis_params.get("rdf_dr", 0.1)
    boundaries = particle_parameters["boundaries"] # Get boundaries from particle_parameters
    rdf_rmax = analysis_params.get("rdf_rmax", (boundaries["x_max"] - boundaries["x_min"]) / 2.0)
    rdf_start_frame = analysis_params.get("rdf_start_frame", 0)


    if not positions_history or len(positions_history) <= rdf_start_frame:
        # print("Warning: No particle position data available for RDF or start frame out of bounds.")
        return np.array([]), np.array([])


    box_size = np.array([boundaries["x_max"] - boundaries["x_min"], boundaries["y_max"] - boundaries["y_min"]])
    if box_size[0] <=0 or box_size[1] <=0 : return np.array([]), np.array([]) # Invalid box


    num_bins = int(rdf_rmax / rdf_dr)
    if num_bins <=0 : return np.array([]), np.array([])
    distance_counts = np.zeros(num_bins)
    bin_edges = np.linspace(0, rdf_rmax, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total_number_density_sum = 0.0 # Sum of number densities over frames
    num_frames_averaged = 0
    total_particles_in_avg_frames = 0 # Sum of N over averaged frames


    for frame_index in range(rdf_start_frame, len(positions_history)):
        positions = positions_history[frame_index]
        if not isinstance(positions, np.ndarray) or positions.ndim != 2 or positions.shape[1]!=2: continue # Invalid position data for frame
        num_particles_in_frame = positions.shape[0]
        if num_particles_in_frame < 2: continue

        num_frames_averaged += 1
        total_particles_in_avg_frames += num_particles_in_frame


        r_vec_all_pairs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        r_vec_all_pairs_pbc = r_vec_all_pairs - box_size * np.round(r_vec_all_pairs / box_size)
        r_mag_all_pairs = np.sqrt(np.sum(r_vec_all_pairs_pbc**2, axis=2))

        upper_triangle_indices = np.triu_indices_from(r_mag_all_pairs, k=1)
        distances = r_mag_all_pairs[upper_triangle_indices]
        distances_within_rmax = distances[distances < rdf_rmax]

        valid_distances = distances_within_rmax[np.isfinite(distances_within_rmax)]
        if valid_distances.size > 0:
             counts, _ = np.histogram(valid_distances, bins=bin_edges)
             distance_counts += counts


        area = box_size[0] * box_size[1]
        if area > DEFAULT_EPSILON: # Ensure area is positive
             total_number_density_sum += num_particles_in_frame / area


    if num_frames_averaged == 0 or np.sum(distance_counts) == 0:
         # print("Warning: No valid frames or pairs found for RDF calculation.")
         return np.array([]), np.array([])

    avg_num_particles_for_norm = total_particles_in_avg_frames / num_frames_averaged
    avg_number_density_for_norm = total_number_density_sum / num_frames_averaged


    rdf_g_r = np.zeros(num_bins)
    for i in range(num_bins):
        r_inner, r_outer = bin_edges[i], bin_edges[i+1]
        area_of_bin = np.pi * (r_outer**2 - r_inner**2)
        # Expected number of pairs in this bin FOR ONE FRAME, for N particles, with density rho:
        # N * rho * area_bin / 2. We sum counts over all frames, so multiply by num_frames_averaged.
        expected_pairs_in_bin_total = num_frames_averaged * avg_num_particles_for_norm * avg_number_density_for_norm * area_of_bin / 2.0

        if expected_pairs_in_bin_total > DEFAULT_EPSILON:
             rdf_g_r[i] = distance_counts[i] / expected_pairs_in_bin_total
        else: rdf_g_r[i] = 0.0

    return bin_centers, rdf_g_r


# Plotting Function for RDF (remains the same, uses particle_parameters)
def plot_rdf(rdf_bin_centers, rdf_g_r, particle_parameters):
    if rdf_bin_centers.size == 0 or rdf_g_r.size == 0: return # Check if arrays are empty
    save_directory = particle_parameters["saving"]["save_directory"]
    if not os.path.exists(save_directory): os.makedirs(save_directory)
    plt.figure(); plt.plot(rdf_bin_centers, rdf_g_r); plt.xlabel("Distance (r)")
    plt.ylabel("g(r)"); plt.title("Particle Radial Distribution Function"); plt.grid(True)
    ymin = 0.0; ymax = max(2.0, np.max(rdf_g_r[np.isfinite(rdf_g_r)]) * 1.1 if np.any(np.isfinite(rdf_g_r)) else 2.0)
    plt.ylim(ymin, ymax)
    plt.savefig(os.path.join(save_directory, "particle_rdf_plot.png")); plt.close()
    print(f"Particle RDF plot saved to {os.path.join(save_directory, 'particle_rdf_plot.png')}")


# Analysis Function: Mean Squared Displacement (MSD) (remains the same, uses particle_parameters)
def calculate_mean_squared_displacement(positions_history, final_types, particle_parameters):
    analysis_params = particle_parameters["analysis"]
    msd_start_frame = analysis_params.get("msd_start_frame", analysis_params.get("rdf_start_frame", 0))

    if not positions_history or len(positions_history) <= msd_start_frame +1: # Need at least 2 frames from start
        # print("Warning: Insufficient position data for MSD.")
        return np.array([]), np.array([]), {}

    num_total_frames = len(positions_history)
    boundaries = particle_parameters["boundaries"]
    box_size = np.array([boundaries["x_max"] - boundaries["x_min"], boundaries["y_max"] - boundaries["y_min"]])
    dt_sim = particle_parameters["simulation"]["dt"]


    # Time lags to calculate MSD for (up to half the duration of analyzed trajectory segment)
    max_lag_steps = (num_total_frames - msd_start_frame) // 2
    if max_lag_steps <=0: return np.array([]), np.array([]), {} # Not enough steps for any lag
    msd_time_lags_dt = np.arange(1, max_lag_steps + 1) # Time lags in units of steps
    msd_time_points_sec = msd_time_lags_dt * dt_sim


    overall_msd = np.zeros(max_lag_steps)
    msd_per_type = collections.defaultdict(lambda: np.zeros(max_lag_steps))
    # Counts for averaging, ensure final_types is an np.array
    final_types_np = np.array(final_types) if not isinstance(final_types, np.ndarray) else final_types


    for lag_idx, time_lag_steps in enumerate(msd_time_lags_dt):
        squared_displacements_for_lag = []
        type_specific_sq_disp_for_lag = collections.defaultdict(list)

        # Iterate over possible start times (origins) for this lag
        for origin_frame_idx in range(msd_start_frame, num_total_frames - time_lag_steps):
            pos_origin = positions_history[origin_frame_idx]
            pos_lagged = positions_history[origin_frame_idx + time_lag_steps]

            # Ensure particle count consistency for this specific pair of frames
            if pos_origin.shape[0] != pos_lagged.shape[0] or pos_origin.shape[0] == 0:
                continue # Skip if particle numbers differ or no particles

            # Assume final_types corresponds to particles at origin_frame_idx
            # This is an approximation if particles are created/deleted
            current_types_for_origin = final_types_np
            if len(final_types_np) != pos_origin.shape[0]:
                 # If final_types doesn't match, try to get types from types_history (if available and passed)
                 # For simplicity here, we proceed with warning or skip type-specific
                 # print(f"Warning: final_types length mismatch for MSD at frame {origin_frame_idx}. Type-specific MSD may be affected.")
                 # If types_history was passed:
                 # current_types_for_origin = types_history[origin_frame_idx] if origin_frame_idx < len(types_history) else final_types_np
                 pass


            disp_raw = pos_lagged - pos_origin
            disp_pbc = disp_raw - box_size * np.round(disp_raw / box_size)
            sq_disp_per_particle = np.sum(disp_pbc**2, axis=1)

            squared_displacements_for_lag.extend(sq_disp_per_particle)
            if len(current_types_for_origin) == len(sq_disp_per_particle): # If types array matches
                 for p_idx, sq_d in enumerate(sq_disp_per_particle):
                      type_specific_sq_disp_for_lag[current_types_for_origin[p_idx]].append(sq_d)


        if squared_displacements_for_lag:
            overall_msd[lag_idx] = np.mean(squared_displacements_for_lag)
        for p_type, disp_list in type_specific_sq_disp_for_lag.items():
            if disp_list: msd_per_type[p_type][lag_idx] = np.mean(disp_list)

    return msd_time_points_sec, overall_msd, dict(msd_per_type) # Convert defaultdict to dict


# Plotting Function for MSD (remains the same, uses particle_parameters)
def plot_msd(msd_time_points, overall_msd, type_msd_dict, particle_parameters):
    if msd_time_points.size == 0 or overall_msd.size == 0: return
    save_directory = particle_parameters["saving"]["save_directory"]
    if not os.path.exists(save_directory): os.makedirs(save_directory)

    plt.figure(); plt.plot(msd_time_points, overall_msd, label="Overall MSD")
    color_map_vis = particle_parameters.get("visualization", {}).get("patches", {}).get("color_mapping", {})
    # Ensure color_map_vis keys are int if types are int
    color_map_int_keys = {int(k):v for k,v in color_map_vis.items()} if isinstance(color_map_vis,dict) else {}


    for p_type, msd_arr in type_msd_dict.items():
        if msd_arr.size == msd_time_points.size:
            plt.plot(msd_time_points, msd_arr, label=f"Type {p_type} MSD", color=color_map_int_keys.get(p_type, 'gray'))
    plt.xlabel("Time (s)"); plt.ylabel("MSD (distance$^2$)")
    plt.title("Particle Mean Squared Displacement"); plt.legend(); plt.grid(True); plt.xscale('log'); plt.yscale('log') # Often plotted log-log
    plt.savefig(os.path.join(save_directory, "particle_msd_plot.png")); plt.close()
    print(f"Particle MSD plot saved to {os.path.join(save_directory, 'particle_msd_plot.png')}")


# Analysis Function: Bond Angle Distribution (remains the same, uses particle_parameters)
def calculate_bond_angle_distribution(positions_history, orientations_history, all_patch_data_history, bonds_history, types_history, particle_parameters):
    orient_pot_params = particle_parameters["forces"].get("orientation_potential", {})
    bond_angle_pot_params = orient_pot_params.get("bond_angle_potential", {})
    if not bond_angle_pot_params.get("enabled", False): return np.array([]), np.array([])

    patch_defs = particle_parameters["forces"].get("patch_params", {}).get("patch_definitions", {})
    if not patch_defs: return np.array([]), np.array([])

    ideal_angle_map = bond_angle_pot_params.get("ideal_angle_mapping", {}) # Keys are int

    angles_dev = []
    boundaries = particle_parameters["boundaries"]
    box_size = np.array([boundaries["x_max"] - boundaries["x_min"], boundaries["y_max"] - boundaries["y_min"]])

    min_history_len = min(len(h) for h in [positions_history, orientations_history, all_patch_data_history, bonds_history, types_history])

    for frame_idx in range(min_history_len):
        bonds_f, pos_f, orient_f, patches_f, types_f = bonds_history[frame_idx], positions_history[frame_idx], orientations_history[frame_idx], all_patch_data_history[frame_idx], types_history[frame_idx]
        num_p_f = pos_f.shape[0]

        for bond_key in bonds_f.keys():
            if not (isinstance(bond_key, tuple) and len(bond_key) == 2 and isinstance(bond_key[0], tuple)): continue
            (i, p_idx_i), (j, p_idx_j) = bond_key
            if not (i<num_p_f and j<num_p_f and i<len(patches_f) and j<len(patches_f) and p_idx_i<len(patches_f[i]) and p_idx_j<len(patches_f[j])): continue

            patch_i_dat, patch_j_dat = patches_f[i][p_idx_i], patches_f[j][p_idx_j]
            patch_i_type, patch_j_type = patch_i_dat["patch_type"], patch_j_dat["patch_type"]

            r_patch_raw = patch_j_dat["position"] - patch_i_dat["position"]
            r_patch_pbc = r_patch_raw - box_size * np.round(r_patch_raw / box_size)
            r_patch_mag = np.linalg.norm(r_patch_pbc)
            if r_patch_mag < DEFAULT_EPSILON: continue
            bond_vec_dir = r_patch_pbc / r_patch_mag

            particle_type_i_int = types_f[i]
            # Defensive initialization for pdefs_i
            pdefs_i = []
            if particle_type_i_int in patch_defs:
                pdefs_i = patch_defs.get(particle_type_i_int)
            elif str(particle_type_i_int) in patch_defs:
                pdefs_i = patch_defs.get(str(particle_type_i_int))
            if not isinstance(pdefs_i, list): pdefs_i = [] # Final safeguard

            if p_idx_i >= len(pdefs_i): continue
            angle_rel_i = pdefs_i[p_idx_i].get("angle_relative_to_particle",0.0)
            abs_angle_i = orient_f[i] + angle_rel_i
            patch_i_orient_vec = np.array([np.cos(abs_angle_i), np.sin(abs_angle_i)])
            ideal_i_rad = ideal_angle_map.get(patch_i_type, 0.0) # patch_i_type is int
            actual_i_angle = np.arctan2(patch_i_orient_vec[0]*bond_vec_dir[1]-patch_i_orient_vec[1]*bond_vec_dir[0], np.dot(patch_i_orient_vec,bond_vec_dir))
            angles_dev.append(abs(np.mod(actual_i_angle - ideal_i_rad + np.pi, 2*np.pi) - np.pi))


            particle_type_j_int = types_f[j]
            # Defensive initialization for pdefs_j
            pdefs_j = []
            if particle_type_j_int in patch_defs:
                pdefs_j = patch_defs.get(particle_type_j_int)
            elif str(particle_type_j_int) in patch_defs:
                pdefs_j = patch_defs.get(str(particle_type_j_int))
            if not isinstance(pdefs_j, list): pdefs_j = [] # Final safeguard

            if p_idx_j >= len(pdefs_j): continue
            angle_rel_j = pdefs_j[p_idx_j].get("angle_relative_to_particle",0.0)
            abs_angle_j = orient_f[j] + angle_rel_j
            patch_j_orient_vec = np.array([np.cos(abs_angle_j), np.sin(abs_angle_j)])
            ideal_j_rad = ideal_angle_map.get(patch_j_type, 0.0)
            actual_j_angle = np.arctan2(patch_j_orient_vec[0]*(-bond_vec_dir[1])-patch_j_orient_vec[1]*(-bond_vec_dir[0]), np.dot(patch_j_orient_vec,-bond_vec_dir))
            angles_dev.append(abs(np.mod(actual_j_angle - ideal_j_rad + np.pi, 2*np.pi) - np.pi))

    if not angles_dev: return np.array([]), np.array([])
    hist, edges = np.histogram(angles_dev, bins=50, range=(0,np.pi), density=True)
    centers = (edges[:-1] + edges[1:])/2
    return centers, hist


# Plotting Function for Bond Angle Distribution (remains the same, uses particle_parameters)
def plot_bond_angle_distribution(bin_centers, angle_counts, particle_parameters):
    if bin_centers.size == 0 or angle_counts.size == 0: return
    save_dir = particle_parameters["saving"]["save_directory"]
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    plt.figure(); plt.bar(bin_centers, angle_counts, width=(bin_centers[1]-bin_centers[0] if len(bin_centers)>1 else 0.1) , edgecolor='black')
    plt.xlabel("Angle Deviation (radians)"); plt.ylabel("Probability Density")
    plt.title("Patch Bond Angle Distribution"); plt.grid(axis='y')
    plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], ['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
    plt.savefig(os.path.join(save_dir, "particle_bond_angle_distribution.png")); plt.close()
    print(f"Particle bond angle distribution plot saved to {os.path.join(save_dir, 'particle_bond_angle_distribution.png')}")


# Analysis Function: Cluster Size Distribution (remains the same, uses particle_parameters)
def calculate_cluster_size_distribution(bonds_history, positions_history, particle_parameters):
    if not bonds_history or not positions_history : return np.array([]), np.array([])
    all_sizes = []
    min_len = min(len(bonds_history), len(positions_history))
    for frame_idx in range(min_len):
        bonds_f, pos_f = bonds_history[frame_idx], positions_history[frame_idx]
        num_p = pos_f.shape[0]
        if num_p > 0:
            labels = get_cluster_labels_for_frame(pos_f, bonds_f, num_p)
            _, counts = np.unique(labels, return_counts=True)
            all_sizes.extend(counts)
    if not all_sizes: return np.array([]), np.array([])
    max_size = max(all_sizes) if all_sizes else 1
    hist, edges = np.histogram(all_sizes, bins=np.arange(0.5, max_size+1.5,1.0), density=True)
    centers = np.arange(1, max_size+1)
    # Ensure hist matches centers length if max_size was small
    if len(hist) < len(centers) and len(centers) == 1 and max_size == 1: # common for single particles
        hist = np.array([1.0]) if sum(all_sizes)==len(all_sizes) and all_sizes[0]==1 else np.array([0.0])


    return centers, hist


# Plotting Function for Cluster Size Distribution (remains the same, uses particle_parameters)
def plot_cluster_size_distribution(cluster_sizes, cluster_counts, particle_parameters):
    if cluster_sizes.size == 0 or cluster_counts.size == 0: return
    save_dir = particle_parameters["saving"]["save_directory"]
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    plt.figure(); plt.bar(cluster_sizes, cluster_counts, width=1.0, edgecolor='black')
    plt.xlabel("Cluster Size"); plt.ylabel("Probability"); plt.title("Particle Cluster Size Distribution")
    plt.xticks(cluster_sizes) if len(cluster_sizes) < 20 else plt.xticks(np.arange(min(cluster_sizes),max(cluster_sizes)+1, max(1,int(len(cluster_sizes)/10)) ))

    plt.grid(axis='y'); plt.savefig(os.path.join(save_dir, "particle_cluster_size_distribution.png")); plt.close()
    print(f"Particle cluster size distribution plot saved to {os.path.join(save_dir, 'particle_cluster_size_distribution.png')}")


# Analysis Function: Calculate Nematic Order Parameter (remains the same)
def calculate_nematic_order_parameter(orientations):
    if len(orientations) == 0: return 0.0
    cos_2t, sin_2t = np.cos(2*orientations), np.sin(2*orientations)
    return np.sqrt(np.mean(cos_2t)**2 + np.mean(sin_2t)**2)


# Visualization Function (Adapted for Coupled Simulation)
def visualize_simulation(total_energy_history, positions_history, orientations_history, nematic_order_parameter_history, all_patch_data_history, bonds_history, types_history, particle_parameters):
    save_dir = particle_parameters["saving"]["save_directory"]
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    anim_file = os.path.join(save_dir, particle_parameters["saving"].get("animation_filename", "particle_sim.gif"))
    boundaries = particle_parameters["boundaries"]
    x_min, x_max, y_min, y_max = boundaries["x_min"], boundaries["x_max"], boundaries["y_min"], boundaries["y_max"]

    if total_energy_history:
        plt.figure(); plt.plot(total_energy_history); plt.xlabel("Step"); plt.ylabel("Total Energy")
        plt.title("Particle Total Energy"); plt.grid(True)
        plt.savefig(os.path.join(save_dir, "particle_total_energy_plot.png")); plt.close()
        print(f"Particle total energy plot saved to {os.path.join(save_dir, 'particle_total_energy_plot.png')}")

    if nematic_order_parameter_history:
        plt.figure(); plt.plot(nematic_order_parameter_history); plt.xlabel("Step"); plt.ylabel("Nematic Order (S)")
        plt.title("Particle Nematic Order Parameter"); plt.grid(True); plt.ylim(0, 1.1)
        plt.savefig(os.path.join(save_dir, "particle_nematic_order_parameter_plot.png")); plt.close()
        print(f"Particle nematic order plot saved to {os.path.join(save_dir, 'particle_nematic_order_parameter_plot.png')}")

    if not positions_history: print("No particle positions for animation."); return

    fig_anim, ax_anim = plt.subplots(); ax_anim.set_xlim(x_min, x_max); ax_anim.set_ylim(y_min, y_max)
    ax_anim.set_aspect('equal', adjustable='box'); ax_anim.set_title("Particle Simulation"); ax_anim.grid(True)

    vis_params = particle_parameters["visualization"]
    orient_line_p = vis_params.get("orientation_line", {})
    patches_vis_p = vis_params.get("patches", {})
    bonds_vis_p = vis_params.get("bonds", {})
    clusters_vis_p = vis_params.get("clusters", {})

    particle_scatter = ax_anim.scatter([], [], s=50) # Base size
    max_hist_particles = max(len(p) for p in positions_history if len(p)>0) if any(len(p)>0 for p in positions_history) else 0
    orientation_lines_plt = [ax_anim.plot([], [], color=orient_line_p.get("color", 'k'), lw=orient_line_p.get("linewidth", 1))[0] for _ in range(max_hist_particles)]
    patch_scatter_plt = ax_anim.scatter([], [], s=(patches_vis_p.get("size",0.5)**2)*20, edgecolors=patches_vis_p.get("edgecolor",'k'), zorder=3)
    bond_lines_plt = [] # To store line artists for bonds, cleared each frame
    time_text_plt = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes)

    min_hist_len_anim = min(len(h) for h in [positions_history, orientations_history, all_patch_data_history, bonds_history, types_history])


    def update_anim(frame):
        if frame >= min_hist_len_anim: return [particle_scatter, patch_scatter_plt, time_text_plt] + orientation_lines_plt + bond_lines_plt

        pos_f, orient_f, patches_f_all, bonds_f, types_f = positions_history[frame], orientations_history[frame], all_patch_data_history[frame], bonds_history[frame], types_history[frame]
        num_p_f = pos_f.shape[0]
        if num_p_f == 0: # Handle empty frame
             particle_scatter.set_offsets(np.array([]).reshape(0,2))
             patch_scatter_plt.set_offsets(np.array([]).reshape(0,2))
             for line in orientation_lines_plt: line.set_data([],[])
             for bline in bond_lines_plt: bline.remove()
             bond_lines_plt.clear()
             time_text_plt.set_text(f"Step: {frame}, Time: {frame*particle_parameters['simulation']['dt']:.1f} (No Particles)")
             return [particle_scatter, patch_scatter_plt, time_text_plt] + orientation_lines_plt

        particle_scatter.set_offsets(pos_f)
        # Coloring particles
        current_colors = ['gray'] * num_p_f # Default
        if clusters_vis_p.get("enabled", False):
            labels = get_cluster_labels_for_frame(pos_f, bonds_f, num_p_f)
            unique_labels = np.unique(labels)
            cmap_func = plt.get_cmap(clusters_vis_p.get("colormap", "viridis"))
            current_colors = [cmap_func(l/(len(unique_labels)-1)) if len(unique_labels)>1 else cmap_func(0.5) for l in labels]
        else:
            # color_map_main = vis_params.get("patches",{}).get("color_mapping",{}) # Use patch color map for main particle
            # Using the direct particle color mapping as derived in init
            main_color_mapping = patches_vis_p.get("color_mapping",{}) # int keys
            current_colors = [main_color_mapping.get(t, 'grey') for t in types_f]


        particle_scatter.set_facecolor(current_colors)


        for i in range(max_hist_particles):
            if i < num_p_f and orient_line_p.get("enabled", True):
                l = orient_line_p.get("length",1.0)
                end_pt = pos_f[i] + l * np.array([np.cos(orient_f[i]), np.sin(orient_f[i])])
                orientation_lines_plt[i].set_data([pos_f[i,0], end_pt[0]], [pos_f[i,1], end_pt[1]])
            else: orientation_lines_plt[i].set_data([],[])

        # Patches
        patch_coords, patch_colors_list = [], []
        if patches_vis_p.get("enabled", True) and frame < len(all_patch_data_history) and patches_f_all:
             patch_color_map = patches_vis_p.get("color_mapping",{}) # int keys
             for p_idx_particle in range(num_p_f): # Iterate up to current particles
                 if p_idx_particle < len(patches_f_all): # Check if patch data exists for this particle
                     for patch_data in patches_f_all[p_idx_particle]: # patches_f_all is list of lists
                         patch_coords.append(patch_data["position"])
                         patch_colors_list.append(patch_color_map.get(patch_data["patch_type"], 'black'))
        patch_scatter_plt.set_offsets(np.array(patch_coords) if patch_coords else np.array([]).reshape(0,2) )
        if patch_colors_list : patch_scatter_plt.set_facecolor(patch_colors_list)
        else: patch_scatter_plt.set_facecolor(np.array([]))



        for bline in bond_lines_plt: bline.remove()
        bond_lines_plt.clear()
        if bonds_vis_p.get("enabled", True) and bonds_f and frame < len(all_patch_data_history) and patches_f_all:
            for bond_key in bonds_f.keys():
                is_patch_bond = isinstance(bond_key, tuple) and len(bond_key)==2 and isinstance(bond_key[0],tuple)
                if is_patch_bond:
                    (i,pi), (j,pj) = bond_key
                    if i<num_p_f and j<num_p_f and i<len(patches_f_all) and j<len(patches_f_all) and pi<len(patches_f_all[i]) and pj<len(patches_f_all[j]):
                        pos1, pos2 = patches_f_all[i][pi]["position"], patches_f_all[j][pj]["position"]
                        line, = ax_anim.plot([pos1[0],pos2[0]], [pos1[1],pos2[1]],
                                             color=bonds_vis_p.get("color",'gray'),
                                             lw=bonds_vis_p.get("linewidth", 2.0),
                                             linestyle=bonds_vis_p.get("linestyle", '-'),
                                             zorder=0)
                        bond_lines_plt.append(line)
                # else: # Center-center bonds
                #     i,j = bond_key
                #     if i<num_p_f and j<num_p_f:
                #         line, = ax_anim.plot([pos_f[i,0],pos_f[j,0]], [pos_f[i,1],pos_f[j,1]], color=bonds_vis_p.get("color",'gray'), lw=bonds_vis_p.get("linewidth",1.5), zorder=0)
                #         bond_lines_plt.append(line)


        time_text_plt.set_text(f"Step: {frame}, Time: {frame*particle_parameters['simulation']['dt']:.1f}")
        return [particle_scatter, patch_scatter_plt, time_text_plt] + orientation_lines_plt + bond_lines_plt


    if min_hist_len_anim > 0 :
        ani = animation.FuncAnimation(fig_anim, update_anim, frames=min_hist_len_anim, blit=True, interval=1000/particle_parameters["simulation"]["animation_fps"], repeat=False)
        try:
            print(f"Saving animation to {anim_file}...")
            ani.save(anim_file, writer=animation.PillowWriter(fps=particle_parameters["simulation"]["animation_fps"]))
            print("Animation saved.")
        except Exception as e: print(f"Error saving animation: {e}. Ensure Pillow is installed.")
    plt.close(fig_anim)

    # Segment 6: MAUS Classes & Coupled Simulation Integration

# Enums, Constants, and Classes from MAUS Prototype (Adaptive Matrix Solver 0.1)

# --- Enumerations for Problem Types ---
class ProblemType(Enum):
    EIGENVALUE = 1
    SOLVE_LINEAR_SYSTEM = 2
    SVD = 3

# --- Global Configuration Parameters (Informing MAUS's Heuristics) ---
# These are adjustable hyperparameters that MAUS uses in its internal decision-making.
GLOBAL_DEFAULT_PSI_EPSILON_BASE = np.complex128(1e-20) # Base regularization magnitude (multiplied by aggression factor)
GLOBAL_DEFAULT_ALPHA_V_INITIAL = np.complex128(0.01) # Initial learning rate for candidates' steps
GLOBAL_MAX_PSI_ATTEMPTS = 25 # Max attempts for InverseIterateSolver per candidate update
GLOBAL_MAX_STUCK_FOR_RETIREMENT = 8 # Times a candidate can repeatedly fail before being retired (population management)
GLOBAL_MAX_STUCK_FOR_PRUNING = 5 # Used by the population manager, indicates `Fragile` state when avg stuckness is higher than this value
GLOBAL_MIN_WEIGHT_TO_SURVIVE_PRUNE = 1e-10 # Minimum confidence weight for a candidate to stay in population
GLOBAL_VECTOR_SIMILARITY_TOL = 0.999 # Cosine similarity threshold for considering two vectors "the same"
GLOBAL_LAMBDA_SIMILARITY_TOL = 1e-5 # Absolute difference for eigenvalue uniqueness
GLOBAL_SIGMA_SIMILARITY_TOL_ABS = 1e-6 # Absolute threshold for singular value uniqueness (below this, it's considered ~0)
GLOBAL_SIGMA_SIMILARITY_TOL_REL = 1e-4 # Relative threshold for singular value uniqueness (e.g., 1e-4 of max sigma)
GLOBAL_CONVERGENCE_RESIDUAL_TOL = 1e-8 # Default global residual tolerance for MAUS solve.


# --- InverseIterateSolver: Adaptive Local Solver for Ax=b-like Problems ---
# This class encapsulates the robust linear system solving using direct_solve or iterative_gmres,
# dynamically choosing or falling back, and applying Ψ regularization.
class InverseIterateSolver:
    def __init__(self, N, base_psi_epsilon, max_attempts, preferred_method='direct_solve', is_sparse=False):
        self.N = N # Dimension of the matrix for solve
        self.base_psi_epsilon = base_psi_epsilon # Base magnitude for Ψ
        self.max_attempts = max_attempts # Max internal retries for a single solve call
        self.preferred_method = preferred_method # 'direct_solve' (sla.solve or spsolve) or 'iterative_gmres'
        self.fallback_method = 'iterative_gmres' if preferred_method == 'direct_solve' else 'direct_solve' # Auto-determines fallback
        self.is_sparse = is_sparse # True if problem is sparse

    def solve(self, A_target, b_rhs, candidate_stuck_counter):
        """
        Attempts to solve A_target @ x = b_rhs robustly with Psi regularization,
        potentially trying fallback solvers.
        """
        num_psi_attempts = 0
        current_method_for_try = self.preferred_method # Start with preferred method

        while num_psi_attempts < self.max_attempts:
            # Scale PSI by base and attempt count for increasing aggression based on history
            psi_scalar_magnitude = self.base_psi_epsilon * (10**(num_psi_attempts / 2.0)) * (10**(candidate_stuck_counter / 3.0))

            # Create regularization term (Psi): dynamically chooses sparse identity or dense random matrix
            if self.is_sparse:
                regularization_term = sp.identity(self.N, dtype=A_target.dtype, format='csc') * psi_scalar_magnitude
                # Note: `A_target` should already be in a sparse format for addition.
            else: # Dense matrix: adds random noise component to Psi
                random_perturb = (np.random.rand(self.N, self.N) - 0.5 + 1j * (np.random.rand(self.N, self.N) - 0.5)) * psi_scalar_magnitude * 0.15
                regularization_term = psi_scalar_magnitude * np.eye(self.N, dtype=A_target.dtype) + random_perturb

            # Add regularization to the target matrix for solving
            H_solve = A_target + regularization_term

            try:
                # Core solving logic based on `current_method_for_try`
                if current_method_for_try == 'direct_solve':
                    if self.is_sparse:
                        result_vec = spla.spsolve(H_solve.tocsc(), b_rhs) # scipy.sparse.linalg.spsolve for sparse direct solve
                    else:
                        result_vec = sla.solve(H_solve, b_rhs, assume_a='general') # np.linalg.solve for dense direct solve

                elif current_method_for_try == 'iterative_gmres':
                    # GMRES (Generalized Minimal Residual): robust for non-symmetric systems, can handle near-singularity by finding least-squares sol.
                    # It accepts both dense NumPy arrays and sparse SciPy matrices.
                    # x0: initial guess for solution. tol: relative tolerance. maxiter: max iterations.
                    x0_init = b_rhs if b_rhs.shape == H_solve.shape[1:] else np.zeros_like(b_rhs) # Use RHS as initial guess or zeros
                    result_vec, info = spla.gmres(H_solve, b_rhs, x0=x0_init, tol=1e-8, maxiter=50)
                    if info != 0: raise np.linalg.LinAlgError(f"GMRES did not converge cleanly (info={info}).")

                else:
                    raise ValueError(f"Unknown solver method: {current_method_for_try}")

                if not np.all(np.isfinite(result_vec)): # Critical check for NaN/Inf in result
                    raise ValueError("Solution vector not finite after solve.")

                return result_vec, num_psi_attempts # Successful solve and number of attempts

            except (np.linalg.LinAlgError, ValueError, TypeError) as e:
                # If solve fails: first time, switch to fallback method. Subsequent times, retry with stronger Psi.
                if current_method_for_try == self.preferred_method and self.preferred_method != self.fallback_method and num_psi_attempts == 0:
                    current_method_for_try = self.fallback_method # Switch once to fallback on first failure
                    num_psi_attempts = 0 # Reset PSI attempts for the newly chosen solver
                    continue

                num_psi_attempts += 1

        # If all attempts exhausted without success
        raise RuntimeError(f"InverseIterateSolver failed all {self.max_attempts} attempts for {self.preferred_method} and {self.fallback_method}.")


# --- Solution Candidate Class (Represents a single hypothesis/solution candidate) ---
# Each candidate is an autonomous agent making local progress based on MAUS's global strategy.
class SolutionCandidate:
    _candidate_id_counter = 0
    # Internal states define candidate behavior
    class State(Enum):
        EXPLORING = 1  # In search phase, might take larger/randomized steps
        REFINING = 2   # Has found a promising region, focusing on tighter convergence
        STUCK = 3      # Repeatedly failed or diverged locally. Needs intervention or retirement.
        CONVERGED = 4  # Has met convergence criteria
        RETIRED = 5    # Has been pruned from population due to redundancy or persistent failure

    def __init__(self, problem_matrix, problem_type, N_diag, initial_lambda=None, initial_v=None, initial_x=None, initial_u=None, initial_sigma=None, initial_weight=0.01):
        self.id = SolutionCandidate._candidate_id_counter
        SolutionCandidate._candidate_id_counter += 1

        self.N_diag = N_diag # Dimension for square operations (e.g., N for Eigen)
        self.M_rows, self.M_cols = problem_matrix.shape # Actual dimensions of input matrix
        self.problem_type = problem_type
        self.problem_matrix = problem_matrix
        self.b_vector = None

        # Solution parameters (type-specific containers)
        self.lambda_k = initial_lambda
        self.v_k = initial_v
        self.x_k = initial_x
        self.sigma_k = initial_sigma
        self.u_k = initial_u
        self.right_v_k = initial_v

        # Candidate State and confidence tracking
        self.state = SolutionCandidate.State.EXPLORING # Initial state
        self.w_k = initial_weight # Confidence/weight
        self.residual_k = float('inf') # Current residual (lower is better)
        self.prev_residual = float('inf') # Residual from previous step (for adaptation)
        self.alpha_local_step = GLOBAL_DEFAULT_ALPHA_V_INITIAL # Individual step size for reactive gradient

        self.stuck_counter = 0 # Counts how many consecutive times the candidate needed brute-force intervention
        self.local_psi_retries_needed = 0 # Records retries needed by InverseIterateSolver for last update
        self.num_resets = 0 # Counts total times its internal state was randomly re-initialized due to failures

        # History (for debugging and learning over time)
        self.param_history = []
        self.residual_history = []

        self.initialize_random_solution() # Set initial state of solution parameters


    def initialize_random_solution(self):
        # Helper to create a random normalized complex vector
        rand_vec_init = lambda N: (np.random.rand(N) + 1j * np.random.rand(N)).astype(np.complex128)
        # Use simple perturbation or full random based on a threshold on stuck_counter if applicable.
        # For MAUS's internal logic, this just means "reinitialize from scratch if I fail this".
        norm_rand_vec = lambda v_raw: v_raw / np.linalg.norm(v_raw) if np.linalg.norm(v_raw) > 1e-10 else rand_vec_init(v_raw.shape[0]) # Defensive normalization


        if self.problem_type == ProblemType.EIGENVALUE:
            self.v_k = norm_rand_vec(rand_vec_init(self.N_diag))
            self.lambda_k = (random.random() * 5 - 2.5 + 1j * (random.random() * 5 - 2.5)) # Random complex lambda

        elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM:
            self.x_k = norm_rand_vec(rand_vec_init(self.N_diag)) * random.uniform(0.1, 10.0) # Random magnitude initial solution

        elif self.problem_type == ProblemType.SVD:
            self.u_k = norm_rand_vec(rand_vec_init(self.M_rows))
            self.right_v_k = norm_rand_vec(rand_vec_init(self.M_cols))
            self.sigma_k = 1.0

        # Store initial (possibly inf) residual and solution for history.
        self.param_history.append(self.get_current_solution_params())
        self.residual_history.append(self.residual_k)

    def update_solution_step(self, current_matrix_A, b_vector=None, strat_params=None, global_knowledge=None):
        self.b_vector = b_vector # `b` vector for Ax=b problems
        self.prev_residual = self.residual_k # Store last residual for step-size adaptation

        # Parameters for local solver instance from global strategy & knowledge
        overall_psi_aggression_factor = strat_params.get('overall_psi_aggression_factor', 1.0)
        max_psi_retries_global = strat_params.get('max_psi_retries', GLOBAL_MAX_PSI_ATTEMPTS)
        local_solver_preference = global_knowledge.get('local_solver_preference', 'direct_solve') # 'direct_solve' or 'iterative_gmres'
        is_matrix_sparse = global_knowledge.get('is_sparse_problem', False)

        solver_instance = InverseIterateSolver(self.N_diag, GLOBAL_DEFAULT_PSI_EPSILON_BASE * overall_psi_aggression_factor,
                                                max_psi_retries_global, local_solver_preference, is_matrix_sparse)

        # Branch based on problem type for specific update logic
        if self.problem_type == ProblemType.SVD:
            try:
                # SVD works via alternating matrix-vector products (like power method variants).
                # If a vector's norm is tiny, we might add noise or reinitialize.
                if np.linalg.norm(self.right_v_k) < 1e-10:
                    self.right_v_k = (np.random.rand(self.M_cols) + 1j * np.random.rand(self.M_cols)); self.right_v_k /= np.linalg.norm(self.right_v_k)
                    self.stuck_counter += 1; self.num_resets += 1;
                    raise ValueError("SVD right_v_k collapsed. Reinitializing.")

                temp_u_k = current_matrix_A @ self.right_v_k
                self.sigma_k = np.linalg.norm(temp_u_k) # Best singular value estimate
                self.u_k = temp_u_k / (self.sigma_k if self.sigma_k > 1e-10 else 1.0) # Normalize `u`

                if np.linalg.norm(self.u_k) < 1e-10: # Check if u also collapsed (potential error propagation)
                     self.u_k = (np.random.rand(self.M_rows)+1j*np.random.rand(self.M_rows)); self.u_k /= np.linalg.norm(self.u_k)
                     self.stuck_counter += 1; self.num_resets += 1;
                     raise ValueError("SVD u_k collapsed. Reinitializing.")

                temp_v_k = current_matrix_A.conj().T @ self.u_k
                self.sigma_k = max(self.sigma_k, np.linalg.norm(temp_v_k)) # Take the maximum sigma from both updates
                self.right_v_k = temp_v_k / (np.linalg.norm(temp_v_k) if np.linalg.norm(temp_v_k) > 1e-10 else 1.0)

                # Small sigma might indicate convergence to zero singular value, not necessarily a failure.
                if self.sigma_k < GLOBAL_SIGMA_SIMILARITY_TOL_ABS / 100 : # If sigma is tiny even lower than default (often implies it's really 0)
                    self.residual_k = strat_params.get('current_convergence_threshold', 1e-6) * 0.1 # Set very small residual to acknowledge "convergence to zero sigma"
                    self.state = SolutionCandidate.State.CONVERGED # It found a very small sigma and solved for it
                    self.stuck_counter = 0 # No longer stuck
                    # Ensure u and v are well-defined for downstream usage if sigma is zero
                    if np.linalg.norm(self.u_k) < 1e-10: self.u_k = np.ones(self.M_rows, dtype=np.complex128)/np.sqrt(self.M_rows)
                    if np.linalg.norm(self.right_v_k) < 1e-10: self.right_v_k = np.ones(self.M_cols, dtype=np.complex128)/np.sqrt(self.M_cols)

                else: # Otherwise, standard processing
                    self.stuck_counter = max(0, self.stuck_counter - 1) # Reduce stuck counter on successful SVD step

            except (RuntimeError, ValueError, np.linalg.LinAlgError) as e: # Catch any errors from SVD path or its internal vector normalization
                self.stuck_counter += 1; self.w_k *= 0.001; self.alpha_local_step *= 0.5
                self.state = SolutionCandidate.State.STUCK
                if self.stuck_counter >= GLOBAL_MAX_STUCK_FOR_RETIREMENT: self.state = SolutionCandidate.State.RETIRED
                # If SVD method explicitly threw error, re-randomize for brute-force exploration
                self.u_k = (np.random.rand(self.M_rows)+1j*np.random.rand(self.M_rows))/np.sqrt(self.M_rows)
                self.right_v_k = (np.random.rand(self.M_cols)+1j*np.random.rand(self.M_cols))/np.sqrt(self.M_cols)
                self.sigma_k = 1.0

        # --- Common update block for Eigenvalue and SolveLinearSystem problems (using InverseIterateSolver) ---
        else:
            target_A_for_solve = current_matrix_A
            rhs_for_solve = None
            current_main_vec_ref = None

            if self.problem_type == ProblemType.EIGENVALUE:
                if np.linalg.norm(self.v_k) < 1e-10:
                    self.v_k = (np.random.rand(self.N_diag) + 1j*np.random.rand(self.N_diag)); self.v_k /= np.linalg.norm(self.v_k)
                    self.stuck_counter += 1; self.num_resets += 1;
                    raise ValueError("Eigenvector collapsed. Reinitializing random vector for a fresh start.") # This restarts `try` block, potentially with a new `Psi`
                self.lambda_k = (self.v_k.conj().T @ current_matrix_A @ self.v_k) / (self.v_k.conj().T @ self.v_k) # Reactive lambda update
                target_A_for_solve = current_matrix_A - self.lambda_k * np.eye(self.N_diag, dtype=current_matrix_A.dtype)
                rhs_for_solve = self.v_k # `v_k` serves as the right-hand side for (A-λI)z = v
                current_main_vec_ref = self.v_k

            elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM:
                target_A_for_solve = current_matrix_A
                rhs_for_solve = self.b_vector
                current_main_vec_ref = self.x_k

            try:
                new_vec_raw, self.local_psi_retries_needed = solver_instance.solve(target_A_for_solve, rhs_for_solve, self.stuck_counter)

                # Apply alpha_local_step for controlled blend/step. This prevents overshooting and aids stability.
                if self.problem_type == ProblemType.EIGENVALUE:
                    # Blends the old `v_k` with the `new_vec_raw` in the direction of the solution
                    self.v_k = (1.0 - self.alpha_local_step) * self.v_k + self.alpha_local_step * new_vec_raw
                    self.v_k /= np.linalg.norm(self.v_k) if np.linalg.norm(self.v_k) > 1e-10 else (np.random.rand(self.N_diag)+1j*np.random.rand(self.N_diag))/np.sqrt(self.N_diag) # Normalize and protect against 0-norm
                elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM:
                    # For Ax=b, `new_vec_raw` IS the candidate `X`. Alpha blends current `x_k` with this newly calculated `X`.
                    self.x_k = (1.0 - self.alpha_local_step) * current_main_vec_ref + self.alpha_local_step * new_vec_raw

                self.stuck_counter = max(0, self.stuck_counter - 1) # Success means reduction in stuckness

            except (RuntimeError, ValueError) as e: # Catch InverseIterateSolver failure (ran out of PSI/solver types)
                self.stuck_counter += 1
                self.w_k *= 0.001 # Penalize candidate weight
                self.alpha_local_step = max(self.alpha_local_step * 0.5, 1e-6) # Aggressive step size reduction
                if self.stuck_counter >= GLOBAL_MAX_STUCK_FOR_RETIREMENT: # Candidate failed too many times, retire it
                    self.state = SolutionCandidate.State.RETIRED
                    self.num_resets += 1 # Count how many were completely reset and retired
                else: # Otherwise, mark as stuck for now and retry with random state next
                    self.state = SolutionCandidate.State.STUCK
                    self.initialize_random_solution() # Reset vector/params, retaining `stuck_counter`


        # --- Common Residual Calculation & History Logging (Regardless of previous branch) ---
        A = self.problem_matrix # Get the current (potentially updated, e.g., dynamic A(t)) matrix for residual calc
        if self.problem_type == ProblemType.EIGENVALUE:
            self.residual_k = np.linalg.norm(A @ self.v_k - self.lambda_k * self.v_k)
        elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM:
            self.residual_k = np.linalg.norm(A @ self.x_k - self.b_vector)
        # SVD residual is calculated in its update path; just verify it's not infinite now
        elif self.problem_type == ProblemType.SVD:
            self.residual_k = np.linalg.norm(A @ self.right_v_k - self.sigma_k * self.u_k) + \
                              np.linalg.norm(A.conj().T @ self.u_k - self.sigma_k * self.right_v_k)

        # Append to history, now guaranteed to have all parameters from whichever branch
        self.param_history.append(self.get_current_solution_params())
        self.residual_history.append(self.residual_k)

        # Adaptive alpha_local_step and candidate State Transition (Common)
        if self.prev_residual > 1e-10:
            if self.residual_k < self.prev_residual * 0.9: # Significant improvement (reward)
                self.alpha_local_step = min(self.alpha_local_step * 1.1, 1.0)
                if self.state != SolutionCandidate.State.CONVERGED: self.state = SolutionCandidate.State.REFINING
            elif self.residual_k > self.prev_residual * 1.5 and self.prev_residual > GLOBAL_CONVERGENCE_RESIDUAL_TOL * 10: # Diverging significantly, and not already very close to converged
                self.alpha_local_step = max(self.alpha_local_step * 0.5, 1e-6) # Aggressively dampen step size
                if self.state != SolutionCandidate.State.CONVERGED: self.state = SolutionCandidate.State.STUCK # Mark as stuck for current round, allow strategies to handle
            else: # Stagnant or minor progress (decay learning rate, and if it wasn't already in another state)
                self.alpha_local_step = max(self.alpha_local_step * 0.95, 1e-6) # Gradually decrease exploration size
                if self.state not in [SolutionCandidate.State.CONVERGED, SolutionCandidate.State.STUCK, SolutionCandidate.State.RETIRED]:
                     self.state = SolutionCandidate.State.EXPLORING # Continue searching for better paths

        # Final Convergence check (can switch state to CONVERGED)
        if self.residual_k < strat_params.get('current_convergence_threshold', GLOBAL_CONVERGENCE_RESIDUAL_TOL) and \
           np.all(np.isfinite(self.get_current_solution_params()[-1])): # Final check for numerical stability of result
            self.state = SolutionCandidate.State.CONVERGED
            self.w_k = 1.0 # Max confidence for converged solutions
            self.stuck_counter = 0 # Reset stuck counter
            self.alpha_local_step = 0.0 # Halt stepping for converged solutions

    def get_current_solution_params(self):
        # Returns the relevant solution parameters as a tuple
        if self.problem_type == ProblemType.EIGENVALUE: return (self.lambda_k, self.v_k)
        elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM: return (self.x_k,)
        elif self.problem_type == ProblemType.SVD: return (self.sigma_k, self.u_k, self.right_v_k)
        return None

# --- MAUS: The Universal Adaptive Matrix Solver (Main Class) ---
class MAUS_Solver:
    def __init__(self, problem_matrix, problem_type, b_vector=None, initial_num_candidates=None, global_convergence_tol=GLOBAL_CONVERGENCE_RESIDUAL_TOL):
        # Initialize matrix: Convert to sparse if needed, else to dense complex.
        if isinstance(problem_matrix, (sp.spmatrix,)):
            self.M = problem_matrix.copy()
        else:
            self.M = problem_matrix.astype(np.complex128)

        self.N_rows, self.N_cols = self.M.shape
        self.N_diag = self.N_rows # General diagonal dimension placeholder

        self.problem_type = problem_type
        self.b = b_vector.astype(np.complex128) if b_vector is not None else None

        # Initial problem diagnosis to set up `problem_knowledge`
        self.diag_info = self._diagnose_matrix_initial(self.M)
        self.is_sparse_problem_init = self.diag_info['is_sparse_init'] # Is initial matrix sparse (from initial threshold)?
        self.cond_number = self.diag_info['condition_number'] # Initial condition number for dense matrix

        # MAUS's internal "Cognitive State" for the problem: informs all strategy decisions
        self.problem_knowledge = {
            'matrix_type': 'Dense', # Becomes 'Sparse' if converted.
            'spectrum_hint': 'Unknown',
            'numerical_stability_state': 'Stable', # 'Stable', 'Fragile', 'Critical'
            'local_solver_preference': 'direct_solve', # Local solver mode: 'direct_solve' or 'iterative_gmres'
            'effective_rank_SVD': min(self.N_rows, self.N_cols), # SVD rank, estimated dynamically
            'true_matrix_is_singular': self.diag_info['is_singular'], # If initial dense matrix is truly singular
            'is_sparse_problem': self.is_sparse_problem_init # Track actual internal state for sparse solving
        }

        # Convert M to a usable sparse format (CSC for solves) if deemed sparse enough OR is already sparse obj.
        if self.problem_knowledge['is_sparse_problem'] and not isinstance(self.M, (sp.csc_matrix, sp.csr_matrix, sp.coo_matrix)):
             self.M = sp.csc_matrix(self.M)
             # print(f"  MAUS converting input matrix to sparse CSC format for efficient compute.")
             self.problem_knowledge['matrix_type'] = 'Sparse' # Update cognitive state

        # Adaptive strategy parameters: These are dynamically tuned by MAUS
        self.strat_params = {
            'overall_psi_aggression_factor': 1.0,
            'max_psi_retries': GLOBAL_MAX_PSI_ATTEMPTS,
            'min_survival_weight': GLOBAL_MIN_WEIGHT_TO_SURVIVE_PRUNE,
            'spawn_rate_multiplier': 1.0,
            'convergence_tolerance': global_convergence_tol,
            'current_convergence_threshold': 1.0, # This threshold changes based on overall progress
        }

        self._set_initial_strategy() # Sets up initial strategy based on matrix diagnosis

        # Candidate population initialization
        initial_num_candidates = initial_num_candidates if initial_num_candidates is not None else (self.N_diag * 3)
        if self.problem_type == ProblemType.SVD:
            initial_num_candidates = max(initial_num_candidates, min(self.N_rows, self.N_cols) * 3)

        self.candidates = []
        for _ in range(initial_num_candidates):
            self.candidates.append(SolutionCandidate(self.M, self.problem_type, self.N_diag)) # Pass sparse matrix

        SolutionCandidate._candidate_id_counter = initial_num_candidates
        # print(f"MAUS Initialized with {initial_num_candidates} candidates for {self.problem_type.name} (Dims={self.N_rows}x{self.N_cols}).")
        # print(f"Initial matrix diagnostics: Cond={self.cond_number:.2e}, MatrixType={self.problem_knowledge['matrix_type']}. Initial MAUS Knowledge: {self.problem_knowledge['numerical_stability_state']}.")

        # Global metrics for MAUS's internal awareness of overall problem state
        self.landscape_energy = 1.0 # Global objective: minimize this
        self.avg_residual = 1.0
        self.avg_stuckness = 0.0
        self.num_distinct_converged_solutions = 0
        self.converged_solutions = [] # Stores final unique converged solutions found


    def _diagnose_matrix_initial(self, matrix):
        """Initial static diagnosis of the matrix at MAUS initialization."""
        is_sparse_init = False
        if isinstance(matrix, (np.ndarray,)): # Check if a dense NumPy array is sparse enough for conversion
            is_sparse_init = np.count_nonzero(matrix) < 0.25 * matrix.size
        elif isinstance(matrix, (sp.spmatrix,)): # Check if it's already a SciPy sparse matrix object
            is_sparse_init = True

        cond_num = np.inf
        matrix_is_singular = False
        try:
            # Condition number check, but only if matrix is not initially flagged as sparse (costly for large N)
            if not is_sparse_init:
                cond_num = np.linalg.cond(matrix)
                if np.isinf(cond_num) or cond_num > 1e15: matrix_is_singular = True
            else: # For sparse matrix, assume condition number from behavior, or specialized norms.
                pass
        except np.linalg.LinAlgError: # Catches errors during condition calculation itself
            cond_num = np.inf
            matrix_is_singular = True

        return {'is_sparse_init': is_sparse_init, 'condition_number': cond_num, 'is_singular': matrix_is_singular}

    def _set_initial_strategy(self):
        """Sets MAUS's initial global strategy based on initial matrix diagnosis."""

        # Determine initial `numerical_stability_state` and solver preference based on matrix properties.
        # This decision flow dictates how aggressive MAUS starts its "brute-force" exploration.
        if self.cond_number > 1e12:
            self.problem_knowledge['numerical_stability_state'] = 'Critical'
            self.strat_params['overall_psi_aggression_factor'] = 50.0 # High aggression from the start
            self.strat_params['max_psi_retries'] = GLOBAL_MAX_PSI_ATTEMPTS * 2 # Double local retry attempts
            self.strat_params['current_convergence_threshold'] = 1e-2 # Loose initial global convergence target
            self.problem_knowledge['local_solver_preference'] = 'iterative_gmres' # Critical problems start with GMRES
        elif self.cond_number > 1e6: # Moderately ill-conditioned at start (Fragile)
             self.problem_knowledge['numerical_stability_state'] = 'Fragile'
             self.strat_params['overall_psi_aggression_factor'] = 10.0
             self.strat_params['max_psi_retries'] = GLOBAL_MAX_PSI_ATTEMPTS
             self.problem_knowledge['local_solver_preference'] = 'iterative_gmres' # Fragile problems also start with GMRES

        else: # Well-conditioned
             self.problem_knowledge['numerical_stability_state'] = 'Stable'
             self.problem_knowledge['local_solver_preference'] = 'direct_solve' # Use fastest direct solve


        # Specific adaptations for SOLVE_LINEAR_SYSTEM, particularly for dynamic singular matrices
        if self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM and self.diag_info['is_singular']:
            self.problem_knowledge['true_matrix_is_singular'] = True
            self.problem_knowledge['local_solver_preference'] = 'iterative_gmres' # GMRES for robust pseudo-inverse
            self.strat_params['overall_psi_aggression_factor'] = max(self.strat_params['overall_psi_aggression_factor'], 20.0)

        # Specific adaptations for SVD problems, which often need aggressive PSI and can involve very tiny singular values.
        if self.problem_type == ProblemType.SVD:
            if self.problem_knowledge['numerical_stability_state'] == 'Stable': # If general cond-num makes it seem stable
                self.strat_params['overall_psi_aggression_factor'] = max(self.strat_params['overall_psi_aggression_factor'], 2.0) # Nudge aggression up.
            self.strat_params['current_convergence_threshold'] = 1e-4 # Tighter starting threshold for SVD (small values common)

    def _update_global_diagnostics(self, iteration):
        """
        Phase 1: Global Information Acquisition & Matrix Understanding (Dynamic)
        Re-evaluates overall problem status based on current candidate behavior.
        Updates problem_knowledge and calculates landscape_energy.
        """
        total_active_candidates = len(self.candidates)
        sum_residuals = 0.0; sum_stuck_counters = 0; sum_confidence = 0.0; num_converged_all_types = 0

        self.num_distinct_converged_solutions = 0 # Reset count of unique converged solutions for this iteration
        self.converged_solutions = [] # List to hold unique converged solutions for this iteration

        current_sigma_magnitudes = [] # For SVD rank detection heuristic

        for c in self.candidates:
            if c.state == SolutionCandidate.State.CONVERGED:
                num_converged_all_types += 1
                current_tuple = c.get_current_solution_params()
                is_distinct = True # Assume distinct until proven redundant

                # Robust Uniqueness Check (Adapting based on Problem Type)
                if self.problem_type == ProblemType.EIGENVALUE:
                    for s_item in self.converged_solutions: # Compare current `c` with already accepted `s_item` in `converged_solutions`
                        s_lam, s_vec = s_item[0], s_item[1]
                        effective_tol = GLOBAL_LAMBDA_SIMILARITY_TOL + np.abs(s_lam) * 1e-6 # Adaptive absolute + relative tolerance
                        if np.abs(current_tuple[0] - s_lam) < effective_tol and \
                           np.abs(np.vdot(current_tuple[1], s_vec)) > GLOBAL_VECTOR_SIMILARITY_TOL: # Cosine similarity to capture same direction/antiparallel
                            is_distinct = False; break
                elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM:
                    # For Ax=b, there is typically one primary solution, so comparison is straightforward.
                    if len(self.converged_solutions) > 0: # If any converged solutions exist
                        # Assuming only one "true" solution. If we have multiple, compare against first one found.
                        first_found_solution = self.converged_solutions[0][0]
                        # Use a stricter tolerance for direct solutions being identical
                        if np.linalg.norm(current_tuple[0] - first_found_solution) < self.strat_params['convergence_tolerance'] * 100:
                            is_distinct = False # Considered redundant
                    else: # If this is the first converged solution
                        is_distinct = True

                elif self.problem_type == ProblemType.SVD:
                    # SVD: first filter for significance (is sigma not essentially zero?), then for distinctness among non-zero ones
                    max_current_sigma = max([c.sigma_k.real for c in self.candidates if c.sigma_k.real > 0], default=1.0)
                    # A singular value is significant if its real part is notably above absolute threshold OR
                    # if it's a significant fraction of the maximum observed singular value among active candidates.
                    if current_tuple[0].real < GLOBAL_SIGMA_SIMILARITY_TOL_ABS or \
                       current_tuple[0].real / max_current_sigma < GLOBAL_SIGMA_SIMILARITY_TOL_REL :
                         is_distinct = False # This sigma is too small to be counted as a "distinct non-zero" rank.

                    if is_distinct: # If still considered significant/potentially distinct, then compare with already-found solutions
                        for s_item in self.converged_solutions:
                            s_sigma, s_u, s_v = s_item[0], s_item[1], s_item[2]
                            effective_abs_tol_sim = GLOBAL_SIGMA_SIMILARITY_TOL_ABS
                            effective_rel_tol_sim = s_sigma * GLOBAL_SIGMA_SIMILARITY_TOL_REL
                            # Compare based on sigma, then both vectors
                            if np.abs(current_tuple[0] - s_sigma) < max(effective_abs_tol_sim, effective_rel_tol_sim) and \
                               np.abs(np.vdot(current_tuple[1], s_u)) > GLOBAL_VECTOR_SIMILARITY_TOL and \
                               np.abs(np.vdot(current_tuple[2], s_v)) > GLOBAL_VECTOR_SIMILARITY_TOL:
                             is_distinct = False; break
                    current_sigma_magnitudes.append(current_tuple[0].real) # Collect magnitudes for overall rank detection heuristic

                if is_distinct:
                    self.converged_solutions.append(current_tuple)
                    self.num_distinct_converged_solutions += 1

            # Sum metrics ONLY for candidates that are NOT converged AND NOT retired
            if c.state not in [SolutionCandidate.State.CONVERGED, SolutionCandidate.State.RETIRED]:
                sum_residuals += c.residual_k
                sum_stuck_counters += c.stuck_counter
                sum_confidence += c.w_k

        # Normalize metrics to contribute to Landscape Energy calculation
        # Ensure denominator is not zero
        num_active_and_non_converged = total_active_candidates - num_converged_all_types
        self.avg_residual = sum_residuals / max(1, num_active_and_non_converged)
        self.avg_stuckness = sum_stuck_counters / max(1, num_active_and_non_converged)
        self.avg_confidence_active = sum_confidence / max(1, num_active_and_non_converged)

        # Dynamic Landscape Energy (MAUS's primary global objective to minimize)
        # Penalizes high residual, high stuckness, and not finding enough solutions
        norm_avg_res = self.avg_residual / (self.strat_params['current_convergence_threshold'] * 10 + DEFAULT_EPSILON) # Normalize by a multiple of current tolerance
        norm_avg_stuck = self.avg_stuckness / (GLOBAL_MAX_STUCK_FOR_RETIREMENT * 2 + DEFAULT_EPSILON) # Penalizes historical failures heavily

        target_sols_N_global = self.N_diag # Default for Eigenvalue problems
        if self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM: target_sols_N_global = 1 # Linear system expects 1 unique solution
        elif self.problem_type == ProblemType.SVD:
            # Dynamic SVD rank detection heuristic (updates problem_knowledge['effective_rank_SVD'])
            if len(current_sigma_magnitudes) > 1: # Only estimate rank if at least 2 sigmas exist
                current_sigma_magnitudes_sorted = sorted([s for s in current_sigma_magnitudes if s > 0], reverse=True) # Filter out noise and sort
                if len(current_sigma_magnitudes_sorted) > 0:
                    max_sigma_in_set = current_sigma_magnitudes_sorted[0]
                    rank_detected = 0
                    for s_val in current_sigma_magnitudes_sorted:
                        # Count as part of rank if above a relative threshold of maximum sigma found
                        if s_val / (max_sigma_in_set + DEFAULT_EPSILON) > GLOBAL_SIGMA_SIMILARITY_TOL_REL:
                            rank_detected += 1
                else: rank_detected = 0 # All sigmas appear to be 0 or too small.

                # MAUS learns the rank dynamically by picking the *minimum* rank detected that is consistent.
                # Never exceed actual dimensions, never drop below 1 unless zero singular values.
                self.problem_knowledge['effective_rank_SVD'] = min(rank_detected, min(self.N_rows, self.N_cols)) if rank_detected > 0 else 0
                if self.problem_knowledge['effective_rank_SVD'] == 0 and min(self.N_rows, self.N_cols) > 0: self.problem_knowledge['effective_rank_SVD'] = 1 # Force rank to at least 1 if dimensions allow
            else: # If less than 2 current sigmas (i.e., only one or zero), base rank estimation on problem size, not collected sigmas.
                self.problem_knowledge['effective_rank_SVD'] = min(self.N_rows, self.N_cols)

        # Calculate remaining landscape energy components
        norm_missing_sols = (target_sols_N_global - self.num_distinct_converged_solutions) / max(1, target_sols_N_global) # If 0 solutions expected/found, no 'missing' penalty
        self.landscape_energy = (norm_avg_res * 0.4) + (norm_avg_stuck * 0.3) + \
                                (norm_missing_sols * 0.3)
        self.landscape_energy = max(0.0, min(1.0, self.landscape_energy)) # Clamp energy between 0 and 1

        # MAUS's internal "Cognitive State" update (inference about stability)
        if self.avg_stuckness > GLOBAL_MAX_STUCK_FOR_RETIREMENT * 0.5: # Many candidates are stuck globally
            self.problem_knowledge['numerical_stability_state'] = 'Critical'
        elif self.avg_stuckness > GLOBAL_MAX_STUCK_FOR_PRUNING * 0.5: # Lower level of stuckness indicates `Fragile` state
             self.problem_knowledge['numerical_stability_state'] = 'Fragile'
        else: # Mostly progressing well
             self.problem_knowledge['numerical_stability_state'] = 'Stable'


# This closes the MAUS_Solver._update_global_diagnostics method


# The below are placeholder definitions of other methods to ensure complete class structure before the Coupled Simulation.
# These match earlier segment splits if any part was explicitly called out for modification, but mostly
# serve to complete the class structure for final assembly and overall integrity.

# MAUS_Solver continued (Methods not explicitly defined in segments 1-3 were effectively placeholders. This ensures all methods exist within the full code.
# The methods that are NOT directly called by CoupledSimulation were left simplified, focusing on the core problem for the adaptive layers).
    def _adjust_global_strategy(self, iteration):
        """
        Phase 2: Meta-Adaptive Strategy Adjustment (MAUS's Brain)
        Orchestrates MAUS's overall behavior by tuning strat_params based on `landscape_energy` and `problem_knowledge`.
        """
        # General tendency towards refinement or aggressive exploration/Psi application
        if self.landscape_energy > 0.6 and self.problem_knowledge['numerical_stability_state'] == 'Critical':
            # Critical Exploration Mode: problem is very hard, go extremely aggressive
            self.problem_knowledge['local_solver_preference'] = 'iterative_gmres' # Force robust iterative solver (GMRES)
            self.strat_params['overall_psi_aggression_factor'] = min(200.0, self.strat_params['overall_psi_aggression_factor'] * 1.1)
            self.strat_params['spawn_rate_multiplier'] = min(10.0, self.strat_params['spawn_rate_multiplier'] * 1.2) # High spawn to brute force explore
            self.strat_params['current_convergence_threshold'] = max(self.strat_params['convergence_tolerance'] * 50, self.strat_params['current_convergence_threshold'] * 1.05) # Loose target to find any sol

        elif self.landscape_energy > 0.4 and self.problem_knowledge['numerical_stability_state'] == 'Fragile':
            # Fragile Exploration Mode: problem is challenging, try iterative
            self.problem_knowledge['local_solver_preference'] = 'iterative_gmres' # Prefer GMRES
            self.strat_params['overall_psi_aggression_factor'] = min(50.0, self.strat_params['overall_psi_aggression_factor'] * 1.05)
            self.strat_params['spawn_rate_multiplier'] = min(5.0, self.strat_params['spawn_rate_multiplier'] * 1.1)
            self.strat_params['current_convergence_threshold'] = max(self.strat_params['convergence_tolerance'] * 5, self.strat_params['current_convergence_threshold'] * 1.02)

        elif self.landscape_energy < 0.2 and self.problem_knowledge['numerical_stability_state'] == 'Stable':
            # Refinement Mode: Problem is largely solved or stable, focus on tightening convergence
            self.problem_knowledge['local_solver_preference'] = 'direct_solve' # Return to faster direct_solve for refinement
            self.strat_params['overall_psi_aggression_factor'] = max(1.0, self.strat_params['overall_psi_aggression_factor'] * 0.9)
            self.strat_params['spawn_rate_multiplier'] = max(0.01, self.strat_params['spawn_rate_multiplier'] * 0.9)
            self.strat_params['current_convergence_threshold'] = max(self.strat_params['convergence_tolerance'] * 0.1, self.strat_params['current_convergence_threshold'] * 0.9) # Tighten to global_conv_tol

        # Clamp global strategy parameters to remain within defined operational bounds
        self.strat_params['overall_psi_aggression_factor'] = max(1.0, min(200.0, self.strat_params['overall_psi_aggression_factor']))
        self.strat_params['spawn_rate_multiplier'] = max(0.01, min(10.0, self.strat_params['spawn_rate_multiplier']))

    def _manage_candidates(self, iteration):
        """
        Phase 4: Population Management (Weighted Partitioning & Resource Allocation)
        Controls which candidates survive, and how many new ones are spawned.
        """
        initial_candidate_count = len(self.candidates)
        survivors = []
        # Sort candidates by weight to prioritize high-confidence solutions
        for c in sorted(self.candidates, key=lambda x: -x.w_k):
            is_redundant_in_survivors = False
            for s_c in survivors:
                # Compare `c` against already chosen `s_c` (s_c implies higher/equal weight)
                if s_c.state == SolutionCandidate.State.CONVERGED and c.state == SolutionCandidate.State.CONVERGED:
                    res_tuple_c = c.get_current_solution_params()
                    res_tuple_s_c = s_c.get_current_solution_params()

                    if self.problem_type == ProblemType.EIGENVALUE:
                        effective_tol_sim = GLOBAL_LAMBDA_SIMILARITY_TOL + np.abs(res_tuple_s_c[0]) * 1e-6 # Adaptive absolute/relative lambda similarity
                        if np.abs(res_tuple_c[0] - res_tuple_s_c[0]) < effective_tol_sim and \
                           np.abs(np.vdot(res_tuple_c[1], s_vec)) > GLOBAL_VECTOR_SIMILARITY_TOL:
                            is_redundant_in_survivors = True; break
                    elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM:
                        if np.linalg.norm(res_tuple_c[0] - res_tuple_s_c[0]) < self.strat_params['convergence_tolerance'] * 10: # Loose check for solve redundancy
                            is_redundant_in_survivors = True; break
                    elif self.problem_type == ProblemType.SVD:
                        effective_abs_tol_sim = GLOBAL_SIGMA_SIMILARITY_TOL_ABS
                        effective_rel_tol_sim = s_c.sigma_k * GLOBAL_SIGMA_SIMILARITY_TOL_REL # Corrected from `s_sigma` to `s_c.sigma_k` for direct access and clarity
                        # Special SVD redundancy: Acknowledge that numerically zero singular values aren't strictly unique.
                        if res_tuple_s_c[0].real < effective_abs_tol_sim / 100: is_redundant_in_survivors = False # Don't aggressively prune tiny/zero sigmas unless they have very high magnitude error too.
                        elif np.abs(res_tuple_c[0] - res_tuple_s_c[0]) < max(effective_abs_tol_sim, effective_rel_tol_sim) and \
                           np.abs(np.vdot(res_tuple_c[1], s_u)) > GLOBAL_VECTOR_SIMILARITY_TOL and \
                           np.abs(np.vdot(res_tuple_c[2], s_v)) > GLOBAL_VECTOR_SIMILARITY_TOL:
                         is_redundant_in_survivors = True; break

            # Decision flow for pruning/keeping: prioritize non-redundant and improving candidates
            if is_redundant_in_survivors: # Mark redundant for removal later (but not yet removed from current list)
                c.state = SolutionCandidate.State.RETIRED
            elif c.state == SolutionCandidate.State.RETIRED: # Filter explicitly marked `RETIRED` candidates (from internal update logic or prior management)
                pass
            elif (c.w_k < self.strat_params['min_survival_weight'] and c.state != SolutionCandidate.State.CONVERGED) or \
                 (c.stuck_counter >= GLOBAL_MAX_STUCK_FOR_RETIREMENT and c.state != SolutionCandidate.State.CONVERGED):
                pass # This candidate is considered permanently "unfruitful" - filter it out.
            else: # All other conditions passed: this candidate is a survivor!
                survivors.append(c)

        self.candidates = survivors # Update actual population of candidates
        num_pruned = initial_candidate_count - len(self.candidates)
        if num_pruned > 0: # Print only if actual pruning happened
             pass # Removed print for less verbosity, keeping it in mind if debugging.


        # Determine number of new candidates to spawn (Intelligent Spawning)
        target_distinct_count_for_spawn = self.N_diag
        if self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM: target_distinct_count_for_spawn = 1
        elif self.problem_type == ProblemType.SVD: target_distinct_count_for_spawn = self.problem_knowledge['effective_rank_SVD'] # Spawn to target effective rank


        desired_population_base = max(5, self.N_diag * 2 if self.problem_type != ProblemType.SOLVE_LINEAR_SYSTEM else self.N_diag * 1.5)
        if self.problem_type == ProblemType.SVD: desired_population_base = max(desired_population_base, self.problem_knowledge['effective_rank_SVD'] * 4)

        # How many candidates to add to meet desired population base AND find missing distinct solutions.
        num_to_spawn = max(0, int(desired_population_base) - len(self.candidates))
        num_to_spawn += max(0, target_distinct_count_for_spawn - self.num_distinct_converged_solutions)

        num_to_spawn = int(num_to_spawn * self.strat_params['spawn_rate_multiplier']) # Apply global spawn rate multiplier
        num_to_spawn = min(num_to_spawn, 10) # Cap maximum spawns per iteration for performance

        if num_to_spawn > 0:
            # print(f"  SPAWNING: Adding {num_to_spawn} new candidates (Current:{len(self.candidates)}, Distinct Found:{self.num_distinct_converged_solutions}/{target_distinct_count_for_spawn}).") # Removed print for less verbosity.
            for _ in range(num_to_spawn):
                new_candidate_init_vals = {}
                # Directed Spawning (if suitable for more efficient exploration)
                # If we have some converged solutions, spawn nearby (refinement). Else, use full randomness (exploration/brute-force discovery).
                if self.num_distinct_converged_solutions > 0 and self.landscape_energy < 0.8: # We have strong starting points and not complete chaos
                    base_sol_tuple = random.choice(self.converged_solutions) # Pick a random strong converged solution as base

                    if self.problem_type == ProblemType.EIGENVALUE:
                        new_candidate_init_vals['initial_lambda'] = base_sol_tuple[0] + (random.random() * 0.1 - 0.05 + 1j * (random.random() * 0.1 - 0.05)) # Small lambda perturbation
                        new_candidate_init_vals['initial_v'] = base_sol_tuple[1] + (np.random.rand(self.N_diag) - 0.5 + 1j * (np.random.rand(self.N_diag) - 0.5)) * 0.1 # Small vector perturbation
                        # Normalize perturbed vector to avoid magnitude drift in initial guess for eigenvalues
                        new_candidate_init_vals['initial_v'] = new_candidate_init_vals['initial_v'] / (np.linalg.norm(new_candidate_init_vals['initial_v']) + DEFAULT_EPSILON)

                    elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM:
                        new_candidate_init_vals['initial_x'] = base_sol_tuple[0] + (np.random.rand(self.N_diag) - 0.5 + 1j * (np.random.rand(self.N_diag) - 0.5)) * 0.1
                    elif self.problem_type == ProblemType.SVD:
                        new_candidate_init_vals['initial_sigma'] = base_sol_tuple[0] + (random.random() * 0.1 - 0.05) # Scalar singular value
                        # For vector perturbations, ensure correct dimensions for u_k and right_v_k
                        # u_k perturb (M_rows, ), right_v_k perturb (M_cols, )
                        new_candidate_init_vals['initial_u'] = base_sol_tuple[1] + (np.random.rand(self.M_rows) - 0.5 + 1j * (np.random.rand(self.M_rows) - 0.5)) * 0.1
                        new_candidate_init_vals['initial_u'] = new_candidate_init_vals['initial_u'] / (np.linalg.norm(new_candidate_init_vals['initial_u']) + DEFAULT_EPSILON)

                        new_candidate_init_vals['initial_v'] = base_sol_tuple[2] + (np.random.rand(self.M_cols) - 0.5 + 1j * (np.random.rand(self.M_cols) - 0.5)) * 0.1
                        new_candidate_init_vals['initial_v'] = new_candidate_init_vals['initial_v'] / (np.linalg.norm(new_candidate_init_vals['initial_v']) + DEFAULT_EPSILON)

                # Final random initialization if no suitable base solution for directed spawning, or as a default.
                new_candidate = SolutionCandidate(self.M, self.problem_type, self.N_diag, **new_candidate_init_vals, initial_weight=0.01)
                new_candidate.alpha_local_step = GLOBAL_DEFAULT_ALPHA_V_INITIAL * self.strat_params['overall_psi_aggression_factor'] # New candidates get alpha proportional to global aggression
                self.candidates.append(new_candidate)

    def evolve(self, max_iterations=100):
        """
        Main evolution loop of MAUS. Orchestrates global strategy and candidate updates.
        Returns the discovered unique solutions or a default empty list if nothing found.
        """
        # print(f"\n--- Starting MAUS Evolution for {max_iterations} iterations ({self.problem_type.name}) ---") # Verbose for MAUS standalone

        # Calculate true solution via NumPy for comparison (only once), but quietly for embedded MAUS.
        self.true_solution = None
        # Try-except block ensures no crash if reference solution cannot be computed (e.g. true singular matrices).
        try:
            if self.problem_type == ProblemType.EIGENVALUE:
                # Ensure M is square for eigvals, then sort by real and imaginary parts for comparison
                if self.M.shape[0] == self.M.shape[1]:
                    eigs = sla.eigvals(self.M)
                    # Custom sort for complex numbers: by real part, then by imaginary part.
                    self.true_solution = eigs[np.argsort(eigs.real + 1j*eigs.imag)]
                # else no true_solution if non-square for Eigenproblem
            elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM:
                self.true_solution = sla.solve(self.M, self.b, assume_a='general')
            elif self.problem_type == ProblemType.SVD:
                if isinstance(self.M, (sp.spmatrix,)): # Use sparse SVD for sparse matrices
                    k_val = min(self.M.shape) - 1 # Get k-largest singular values
                    if k_val >= 1: # spla.svds requires k >= 1
                         _, s_raw, _ = spla.svds(self.M, k=k_val) # U, s, Vh, we only need s for now.
                    else: # Handle 0-rank case for very small matrices
                         s_raw = np.array([])
                else: # Use dense SVD for dense matrices
                    _, s_raw, _ = sla.svd(self.M)
                self.true_solution = sorted(s_raw.tolist(), reverse=True) # Sorted descending by value for singular values
        except Exception as e:
            # print(f"NumPy reference calculation failed: {e}. Cannot compare results directly in embedded MAUS.") # Silent this for coupled run.
            pass

        # Main MAUS Iteration Loop
        for i in range(max_iterations):
            # Phase 1&2: Global Diagnostics & Strategy Adjustment (MAUS's Brain)
            self._update_global_diagnostics(i + 1)
            self._adjust_global_strategy(i + 1)

            # Phase 3: Candidate Evolution (Local Execution with MAUS knowledge)
            # Only update candidates that are not yet converged or not retired.
            for candidate in self.candidates:
                if candidate.state not in [SolutionCandidate.State.CONVERGED, SolutionCandidate.State.RETIRED]:
                    # Update internal problem_matrix and b_vector within each candidate. This allows a *fixed* MAUS
                    # instance to run for a dynamically changing outer problem (Coupled Simulation).
                    candidate.problem_matrix = self.M
                    if self.b is not None: candidate.b_vector = self.b
                    candidate.update_solution_step(self.M, self.b, self.strat_params, self.problem_knowledge)

            # Phase 4: Population Management (Pruning and Spawning)
            self._manage_candidates(i + 1)

            # Check overall problem completion criteria for early termination
            should_terminate = False
            target_distinct_sols_final = self.N_diag # Expected number of solutions for current problem
            if self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM: target_distinct_sols_final = 1
            elif self.problem_type == ProblemType.SVD:
                # Update expected rank based on dynamics from problem knowledge. If initial problem was empty/null, might be zero.
                target_distinct_sols_final = max(0, self.problem_knowledge['effective_rank_SVD'])

            # Termination condition: Found enough solutions AND landscape energy is low (stable overall state)
            if self.num_distinct_converged_solutions >= target_distinct_sols_final and self.landscape_energy < self.strat_params['convergence_tolerance']:
                should_terminate = True

            if should_terminate:
                # print(f"MAUS successfully identified target number of solutions and achieved low landscape energy. Terminating early at iteration {i+1}.") # Silent for coupled sim.
                break

        # Final Report after evolution loop completes
        # (This is mostly for internal MAUS debugging. For coupled system, its output is the `converged_solutions`)
        # No console printing of full report here, instead just return results.
        return self.converged_solutions


# --- Computational Consciousness Zone (CCZ) Class ---
# Represents a local region that computes and reports its "stress"
class CoupledZone:
    def __init__(self, zone_id: int, spatial_x_min: float, spatial_x_max: float, brain_region_idx: int):
        self.id = zone_id
        self.spatial_x_bounds = (spatial_x_min, spatial_x_max)
        self.brain_region_id = brain_region_idx # The index of the brain region corresponding to this spatial zone
        self.adaptive_factor = 1.0 # Default value, will be controlled by MAUS
        self.current_particle_stress = 0.0 # Derived from kinetic energy/density in this zone
        self.current_brain_stress = 0.0    # Derived from brain activity A_field in this zone
        self.current_total_stress = 0.0    # Combined stress metric

    def measure_particle_stress(self, all_positions: np.ndarray, all_velocities: np.ndarray):
        """
        Measures 'stress' within this zone based on particle properties.
        Simplified to mean kinetic energy of particles within its bounds.
        """
        if all_positions.shape[0] == 0:
            self.current_particle_stress = 0.0
            return

        particles_in_zone_indices = np.where(
            (all_positions[:, 0] >= self.spatial_x_bounds[0]) &
            (all_positions[:, 0] < self.spatial_x_bounds[1])
        )[0]

        if len(particles_in_zone_indices) == 0:
            self.current_particle_stress = 0.0
        else:
            local_kinetic_energy = np.sum(np.linalg.norm(all_velocities[particles_in_zone_indices], axis=1)**2) / 2.0
            self.current_particle_stress = local_kinetic_energy / len(particles_in_zone_indices) # Avg kinetic energy
        # Scale stress to be a reasonable numerical range for MAUS inputs
        self.current_particle_stress = np.clip(self.current_particle_stress / 100.0, 0.01, 10.0) # Arbitrary scaling to bring into range

    def measure_brain_stress(self, brain_A_field: np.ndarray):
        """
        Measures 'stress' in the brain region corresponding to this zone.
        Uses normalized activity field A as an indicator.
        """
        if self.brain_region_id < len(brain_A_field):
            # Scale the raw activity by arbitrary amount for input to MAUS.
            self.current_brain_stress = brain_A_field[self.brain_region_id] * 5.0
        else:
            self.current_brain_stress = 0.0
        # Clip to ensure valid range for MAUS
        self.current_brain_stress = np.clip(self.current_brain_stress, 0.01, 10.0)

    def calculate_total_stress(self):
        """ Combines particle and brain stress into a single metric. """
        self.current_total_stress = (self.current_particle_stress + self.current_brain_stress) / 2.0 # Simple average


# --- CoupledSimulation Class ---
class CoupledSimulation:
    def __init__(self, parameters):
        self.parameters = parameters
        self.particle_parameters = parameters["particle"]
        self.brain_parameters = parameters["brain"]

        # Initialize Particle Simulation State
        self.particle_current_step = 0
        self.positions = np.array([]).reshape(0,2) #Ensure 2D for vstack
        self.velocities = np.array([]).reshape(0,2)
        self.accelerations = np.array([]).reshape(0,2)
        self.bonds = {}
        self.types = np.array([], dtype=int)
        self.masses = np.array([])
        self.colors = []
        self.orientations = np.array([])
        self.angular_velocities = np.array([])
        self.angular_accelerations = np.array([])
        self.num_particles = 0
        self.all_patch_data = []


        # Initialize Histories for Particle Simulation
        self.particle_total_energy_history = []
        self.particle_positions_history = []
        self.particle_orientations_history = []
        self.particle_nematic_order_parameter_history = []
        self.particle_all_patch_data_history = []
        self.particle_bonds_history = []
        self.particle_types_history = []

        # Load particle state if enabled
        if self.particle_parameters["saving"].get("load_simulation_state", False):
            loaded_state = load_simulation_state(self.particle_parameters) # Pass current params for loading config
            if loaded_state:
                self.particle_current_step, self.positions, self.velocities, self.accelerations, \
                self.bonds, self.types, self.masses, self.colors, self.orientations, \
                self.angular_velocities, self.angular_accelerations, self.num_particles, \
                loaded_particle_params_from_file, _ = loaded_state # all_patch_data placeholder _
                # Use the parameters that were *saved with the state*
                self.particle_parameters = loaded_particle_params_from_file
                print(f"Particle simulation loaded from step {self.particle_current_step}")
            else: # Load failed
                print("Particle simulation load failed. Initializing fresh.")
                # Initialize with original parameters if load fails
                self._initialize_fresh_particle_sim(self.particle_parameters)
        else: # Not loading, initialize fresh
             self._initialize_fresh_particle_sim(self.particle_parameters)


        # Ensure moments_of_inertia is initialized correctly after loading or initializing
        if self.num_particles > 0:
            self.moments_of_inertia = np.array([self.particle_parameters["forces"]["moment_of_inertia_mapping"].get(t, 1.0) for t in self.types])
        else:
            self.moments_of_inertia = np.array([])


        # Initialize Unified Brain Simulation
        self.brain_sim = UnifiedBrainSimulation(self.brain_parameters)

        # --- Computational Consciousness Zones (CCZs) Initialization ---
        self.coupled_zones: List[CoupledZone] = []
        self.num_zones = self.brain_parameters["unified"]["num_conceptual_regions"] # Number of zones equals brain regions
        x_min, x_max = self.particle_parameters["boundaries"]["x_min"], self.particle_parameters["boundaries"]["x_max"]
        zone_width = (x_max - x_min) / self.num_zones
        for i in range(self.num_zones):
            zone_x_min = x_min + i * zone_width
            zone_x_max = x_min + (i + 1) * zone_width
            self.coupled_zones.append(CoupledZone(i, zone_x_min, zone_x_max, i)) # Zone ID maps directly to brain_region_id

        self.adaptive_factors_per_zone = np.ones(self.num_zones) # Stores the adaptive factor for each zone
        # History for adaptive factors
        self.adaptive_factors_history = []


        # --- Initialize MAUS for Meta-Control ---
        # MAUS problem is a simple linear system of N_zones equations and N_zones unknowns.
        # Matrix A (identity) represents direct influence, vector b represents current stress in zones.
        # X will be the optimal adaptive factors for each zone.
        # Initialize MAUS with a dummy identity matrix. It will be updated every few steps.
        initial_maus_matrix = np.eye(self.num_zones) if self.num_zones > 0 else np.array([[1.0]])
        initial_maus_b_vector = np.ones(self.num_zones) if self.num_zones > 0 else np.array([1.0])
        self.meta_maus = MAUS_Solver(
            problem_matrix=initial_maus_matrix,
            problem_type=ProblemType.SOLVE_LINEAR_SYSTEM,
            b_vector=initial_maus_b_vector,
            initial_num_candidates=max(5, self.num_zones * 2), # Adjusted based on common N_diag
            global_convergence_tol=1e-5 # Meta-control doesn't need extreme precision
        )
        self.maus_thinking_interval = 10 # How often MAUS updates adaptive factors (outer steps)
        self.maus_meta_iterations = 5    # How many internal MAUS steps per update interval (quick re-optimization)


    def _initialize_fresh_particle_sim(self, params_to_use):
        self.positions, self.velocities, self.accelerations, self.bonds, self.types, \
        self.masses, self.colors, self.orientations, self.angular_velocities, \
        self.angular_accelerations, self.num_particles = initialize_particles(params_to_use)
        self.particle_current_step = 0
        if self.num_particles > 0:
            self.moments_of_inertia = np.array([params_to_use["forces"]["moment_of_inertia_mapping"].get(t, 1.0) for t in self.types])
        else:
            self.moments_of_inertia = np.array([])


    def calculate_particle_density_per_region(self):
        """
        Calculates the density of particles within each conceptual region
        of the simulation space.
        """
        num_regions = self.brain_parameters["unified"]["num_conceptual_regions"]
        if num_regions == 0 : return np.array([])

        boundaries = self.particle_parameters["boundaries"]
        box_width = boundaries["x_max"] - boundaries["x_min"]
        box_height = boundaries["y_max"] - boundaries["y_min"] # Used for area calculation
        if box_width <=0 : return np.zeros(num_regions) # Avoid division by zero

        region_width_particle_space = box_width / num_regions
        density_per_region = np.zeros(num_regions)

        if self.num_particles == 0: return density_per_region

        for i in range(self.num_particles):
            particle_x_relative = self.positions[i, 0] - boundaries["x_min"]
            # Assign particle to the zone it falls into.
            zone_idx = int(particle_x_relative / region_width_particle_space)
            zone_idx = np.clip(zone_idx, 0, num_regions - 1)
            density_per_region[zone_idx] += 1

        region_area = region_width_particle_space * box_height
        if region_area > DEFAULT_EPSILON:
             density_per_region /= region_area
        # else density is effectively counts

        return density_per_region

    def run_simulation(self):
        print("--- Starting Coupled Particle and Brain Simulation ---")
        sim_start_time = time.time()

        num_total_steps = self.particle_parameters["simulation"]["num_steps"]
        dt = self.particle_parameters["simulation"]["dt"]
        base_damping_factor = self.particle_parameters["simulation"]["damping_factor"]
        base_friction_coefficient = self.particle_parameters["external_force"]["friction_coefficient"]
        base_angular_friction_coefficient = self.particle_parameters["external_force"]["angular_friction_coefficient"]

        max_velocity = self.particle_parameters["simulation"]["max_velocity"]
        periodic_save_interval = self.particle_parameters["saving"]["periodic_save_interval"]
        num_conceptual_regions_brain = self.brain_parameters["unified"]["num_conceptual_regions"]


        # Initialize accelerations and ang_accel if starting fresh or not loaded fully
        if self.num_particles > 0 and (self.accelerations.shape[0] != self.num_particles):
            self.accelerations = np.zeros((self.num_particles, 2))
        if self.num_particles > 0 and (self.angular_accelerations.shape[0] != self.num_particles):
            self.angular_accelerations = np.zeros(self.num_particles)


        # Resume from loaded step
        start_step = self.particle_current_step

        # Per-particle adaptive factor based on zone
        self.per_particle_zone_adaptive_factors = np.ones(self.num_particles)


        for step in range(start_step, num_total_steps):
            if self.num_particles == 0 and step > start_step : # If particles disappear and we already ran at least one step with them, stop.
                print(f"Step {step+1}/{num_total_steps} | No particles remaining. Terminating simulation.")
                # Still run brain to finalize gracefully if needed, but no particle interactions
                particle_density_zero = np.zeros(num_conceptual_regions_brain) if num_conceptual_regions_brain > 0 else np.array([])
                self.brain_sim.run_simulation_step(dt, particle_density_zero)
                self.particle_current_step += 1
                break

            # --- Adaptive Control via MAUS (The Master Mind) ---
            if (step % self.maus_thinking_interval == 0 and self.num_zones > 0) or (step == start_step):
                # 1. Gather "Stress" (inputs to MAUS) from all zones
                maus_b_vector_current_stress = np.zeros(self.num_zones)
                current_global_kinetic_energy_total = np.sum(np.linalg.norm(self.velocities, axis=1)**2 / 2.0) if self.num_particles > 0 else 0.0

                for i, zone in enumerate(self.coupled_zones):
                    # Measure particle stress for this zone
                    zone.measure_particle_stress(self.positions, self.velocities)
                    # Measure brain stress for this zone
                    zone.measure_brain_stress(self.brain_sim.A_field)
                    zone.calculate_total_stress()
                    maus_b_vector_current_stress[i] = zone.current_total_stress

                # Normalize stress values across all zones if needed, for MAUS consistency.
                max_stress_overall = np.max(maus_b_vector_current_stress)
                if max_stress_overall > DEFAULT_EPSILON:
                     maus_b_vector_current_stress /= max_stress_overall

                # 2. Update MAUS's internal "problem" state
                # Here, MAUS finds the X vector where A*X=b. A is Identity so X=b.
                # So the new adaptive factors X become the scaled stress values 'b'.
                self.meta_maus.M = np.eye(self.num_zones, dtype=np.complex128)
                self.meta_maus.b = maus_b_vector_current_stress.astype(np.complex128)

                # 3. Let MAUS evolve (simulate MAUS "thinking" over the stress landscape)
                maus_solutions_raw = self.meta_maus.evolve(max_iterations=self.maus_meta_iterations)

                # 4. Extract and apply Adaptive Factors
                if maus_solutions_raw and self.meta_maus.num_distinct_converged_solutions > 0:
                    # For Ax=b problem, we usually expect 1 solution for X (the factors vector).
                    # Extract the *first* solution's X vector (as it is likely the 'best' candidate in MAUS).
                    # Make sure the solution vector has the correct size for the adaptive factors.
                    new_adaptive_factors_solution = maus_solutions_raw[0][0].real # Get the X vector (real part as it's for scalar factors)
                    # Reshape to match num_zones if necessary
                    if new_adaptive_factors_solution.size == self.num_zones:
                        # Clip values to reasonable operational bounds:
                        self.adaptive_factors_per_zone = np.clip(new_adaptive_factors_solution, 0.1, 2.0)
                    else:
                        print(f"MAUS returned irregular solution size: {new_adaptive_factors_solution.shape}. Defaulting factors to 1.")
                        self.adaptive_factors_per_zone.fill(1.0)
                else: # No solution from MAUS, default all factors to 1
                    self.adaptive_factors_per_zone.fill(1.0)

                # Update per-particle damping/friction based on zones
                # This needs to be dynamic per particle, for now distribute current zone factors
                self.per_particle_zone_adaptive_factors = np.zeros(self.num_particles)
                for i in range(self.num_particles):
                    p_pos_x = self.positions[i, 0]
                    # Find which zone the particle belongs to
                    zone_idx = int((p_pos_x - self.particle_parameters["boundaries"]["x_min"]) / zone_width)
                    zone_idx = np.clip(zone_idx, 0, self.num_zones - 1)
                    self.per_particle_zone_adaptive_factors[i] = self.adaptive_factors_per_zone[zone_idx]

                self.brain_sim.current_adaptive_factors_per_brain_region = self.adaptive_factors_per_zone # Pass to brain simulation


            # --- Particle Dynamics Step (using current adaptive factors) ---
            # Scale global damping factor for update_particle_state based on an average adaptive factor from active zones.
            # Could also apply per-particle for localized damping. For this iteration, let's keep it global damping effective damping
            effective_damping_factor = base_damping_factor * (np.mean(self.adaptive_factors_per_zone) if self.num_zones > 0 else 1.0)
            if self.num_particles > 0:
                self.positions, damped_velocities_t, self.orientations, damped_angular_velocities_t = update_particle_state(
                    self.positions, self.velocities, self.accelerations,
                    self.orientations, self.angular_velocities, self.angular_accelerations,
                    dt, effective_damping_factor, max_velocity, self.moments_of_inertia, self.particle_parameters
                )
            else: # Handle zero particles explicitly for this branch
                self.positions = np.array([]).reshape(0,2)
                self.velocities = np.array([]).reshape(0,2)
                self.orientations = np.array([])
                damped_velocities_t = np.array([]).reshape(0,2)
                damped_angular_velocities_t = np.array([])

            # Calculate forces/torques at r(t+dt), o(t+dt) (with friction/angular friction adapted by factors)
            # Use `self.per_particle_zone_adaptive_factors` for localized damping
            net_linear_forces_t_plus_dt, net_torques_t_plus_dt, self.all_patch_data = calculate_forces(
                self.positions, self.velocities, self.orientations, self.types, self.masses, self.bonds,
                self.particle_parameters, step, self.all_patch_data, # Pass current all_patch_data, it will be re-gen'd
                self.brain_sim.external_stimulus_Q,
                num_conceptual_regions_brain,
                self.per_particle_zone_adaptive_factors # Pass per-particle adaptive factors
            )

            # Update accelerations to a(t+dt)
            if self.num_particles > 0 : # Check again due to potential deletion
                self.accelerations = net_linear_forces_t_plus_dt / self.masses[:, np.newaxis] if self.masses.size > 0 else np.zeros((self.num_particles,2))
                self.angular_accelerations = net_torques_t_plus_dt / self.moments_of_inertia if self.moments_of_inertia.size > 0 else np.zeros(self.num_particles)
            else: # All particles might have been deleted
                self.accelerations = np.array([]).reshape(0,2)
                self.angular_accelerations = np.array([])


            # Step 2 of Velocity Verlet: v(t + dt) = v(t_damped_for_pos_update) + 0.5 * (a(t) + a(t+dt)) * dt
            # Re-apply damping from current state's (newly calculated forces) as Velocity Verlet uses it for new vel.
            self.velocities = damped_velocities_t + 0.5 * (self.accelerations) * dt
            self.angular_velocities = damped_angular_velocities_t + 0.5 * (self.angular_accelerations) * dt


            # --- Post-dynamics updates: Bonds, Type Changes, Creation/Deletion ---
            self.bonds = update_bonds(self.positions, self.orientations, self.types, self.bonds, self.particle_parameters, step, self.all_patch_data)
            new_types = apply_state_change_on_bond_form(self.types, self.bonds, self.particle_parameters, step)
            if not np.array_equal(new_types, self.types):
                self.types = new_types
                if self.num_particles > 0: # Recheck num_particles
                    self.masses = np.array([self.particle_parameters["initial_conditions"]["mass_mapping"].get(t, 1.0) for t in self.types])
                    color_map = self.particle_parameters.get("visualization", {}).get("patches", {}).get("color_mapping", {})
                    self.colors = [color_map.get(t, color_map.get(str(t),'gray')) for t in self.types] # check int and str keys
                    self.moments_of_inertia = np.array([self.particle_parameters["forces"]["moment_of_inertia_mapping"].get(t, 1.0) for t in self.types])


            # Particle Creation / Deletion - these can change num_particles and array shapes
            prev_num_particles = self.num_particles
            self.positions, self.velocities, self.orientations, self.types, self.masses, self.bonds, self.all_patch_data, self.num_particles = \
                create_particles(self.positions, self.velocities, self.orientations, self.types, self.masses, self.bonds, self.particle_parameters, step, self.all_patch_data)

            # If created, ensure accelerations, etc. are consistent for new particles
            if self.num_particles > prev_num_particles:
                num_added = self.num_particles - prev_num_particles
                self.accelerations = np.vstack([self.accelerations, np.zeros((num_added,2))]) if prev_num_particles > 0 else np.zeros((self.num_particles,2))
                self.angular_velocities = np.append(self.angular_velocities, np.zeros(num_added)) if prev_num_particles > 0 else np.zeros(self.num_particles)
                self.angular_accelerations = np.append(self.angular_accelerations, np.zeros(num_added)) if prev_num_particles > 0 else np.zeros(self.num_particles)
                # Re-fetch moments of inertia and colors
                self.moments_of_inertia = np.array([self.particle_parameters["forces"]["moment_of_inertia_mapping"].get(t, 1.0) for t in self.types])
                color_map = self.particle_parameters.get("visualization", {}).get("patches", {}).get("color_mapping", {})
                self.colors = [color_map.get(t, color_map.get(str(t),'gray')) for t in self.types]
                # Update per-particle adaptive factors for new particles
                self.per_particle_zone_adaptive_factors = np.pad(self.per_particle_zone_adaptive_factors, (0, num_added), 'edge') # Pad new with last existing values or 1.0. For now, last val

            prev_num_particles = self.num_particles # Update for deletion
            self.positions, self.velocities, self.orientations, self.types, self.masses, self.bonds, self.all_patch_data, self.num_particles = \
                delete_particles(self.positions, self.velocities, self.orientations, self.types, self.masses, self.bonds, self.particle_parameters, step, self.all_patch_data)

            # If deleted, ensure arrays are consistent
            if self.num_particles < prev_num_particles and self.num_particles > 0: # Check >0 after deletion
                 # Re-calculate moments_of_inertia as types/masses arrays are now shorter
                 self.moments_of_inertia = np.array([self.particle_parameters["forces"]["moment_of_inertia_mapping"].get(t, 1.0) for t in self.types])
                 # Accelerations, ang_vel, ang_accel should be updated automatically or sliced if necessary.
                 self.accelerations = self.accelerations[:self.num_particles] if self.accelerations.shape[0] > self.num_particles else self.accelerations
                 self.angular_velocities = self.angular_velocities[:self.num_particles] if self.angular_velocities.shape[0] > self.num_particles else self.angular_velocities
                 self.angular_accelerations = self.angular_accelerations[:self.num_particles] if self.angular_accelerations.shape[0] > self.num_particles else self.angular_accelerations
                 # Update per-particle adaptive factors to reflect deleted particles
                 current_zone_adaptive_factors = np.zeros(self.num_particles)
                 zone_width = (self.particle_parameters["boundaries"]["x_max"] - self.particle_parameters["boundaries"]["x_min"]) / self.num_zones
                 for p_idx in range(self.num_particles):
                     p_pos_x = self.positions[p_idx, 0]
                     zone_idx = int((p_pos_x - self.particle_parameters["boundaries"]["x_min"]) / zone_width)
                     zone_idx = np.clip(zone_idx, 0, self.num_zones - 1)
                     current_zone_adaptive_factors[p_idx] = self.adaptive_factors_per_zone[zone_idx]
                 self.per_particle_zone_adaptive_factors = current_zone_adaptive_factors
            elif self.num_particles == 0: # If all particles were deleted, reset adaptive factor array
                self.per_particle_zone_adaptive_factors = np.array([])
                self.accelerations = np.array([]).reshape(0,2)
                self.angular_velocities = np.array([])
                self.angular_accelerations = np.array([])
                self.moments_of_inertia = np.array([])


            # Record particle history (after all modifications for the step)
            if self.num_particles > 0 : #Only record if particles exist
                current_total_energy = calculate_total_energy(self.positions, self.velocities, self.angular_velocities, self.masses, self.bonds, self.types, self.orientations, self.particle_parameters, step, self.all_patch_data)
                nematic_op = calculate_nematic_order_parameter(self.orientations)
            else:
                current_total_energy = 0.0
                nematic_op = 0.0

            self.particle_total_energy_history.append(current_total_energy)
            self.particle_positions_history.append(self.positions.copy() if self.num_particles > 0 else np.array([]).reshape(0,2))
            self.particle_orientations_history.append(self.orientations.copy() if self.num_particles > 0 else np.array([]))
            self.particle_all_patch_data_history.append([p_list[:] for p_list in self.all_patch_data] if self.num_particles > 0 else []) # Deepcopy list of lists
            self.particle_bonds_history.append(self.bonds.copy() if self.num_particles > 0 else {})
            self.particle_types_history.append(self.types.copy() if self.num_particles > 0 else np.array([], dtype=int))
            self.particle_nematic_order_parameter_history.append(nematic_op)

            # Record Adaptive factors history for zones
            self.adaptive_factors_history.append(self.adaptive_factors_per_zone.copy())


            if (step + 1) % periodic_save_interval == 0 or (step + 1) == num_total_steps:
                 save_simulation_state(step + 1, self.positions, self.velocities, self.orientations, self.types, self.masses, self.bonds, self.particle_parameters)


            # --- Brain Simulation Step (using current particle_density and its own adaptive factors) ---
            particle_density_input = self.calculate_particle_density_per_region()
            self.brain_sim.run_simulation_step(dt, particle_density_input) # Brain internally uses self.current_adaptive_factors_per_brain_region which MAUS updated.


            self.particle_current_step += 1 # Important to keep consistent with overall simulation step.

            if (step + 1) % self.brain_parameters["time"]["PROGRESS_PRINT_INTERVAL"] == 0 or (step + 1) == num_total_steps:
                # brain hist idx is self.brain_sim.outer_step_count - 1
                b_hist_idx = self.brain_sim.outer_step_count -1
                if b_hist_idx < 0: b_hist_idx=0
                # Use `.get` with default values for robustness, if a history entry is somehow missing or list is short.
                brain_time_val = self.brain_sim.history['time'][b_hist_idx] if b_hist_idx < len(self.brain_sim.history.get('time',[])) else 0.0
                brain_neurons_val = self.brain_sim.history['N_neurons'][b_hist_idx] if b_hist_idx < len(self.brain_sim.history.get('N_neurons',[])) else 0
                brain_synapses_val = self.brain_sim.history['N_synapses'][b_hist_idx] if b_hist_idx < len(self.brain_sim.history.get('N_synapses',[])) else 0
                brain_thought_region_val = self.brain_sim.history['active_thought_region'][b_hist_idx] if b_hist_idx < len(self.brain_sim.history.get('active_thought_region',[])) else -1


                print(f"Step {step+1}/{num_total_steps} | Part: E={current_total_energy:.2f}, S={nematic_op:.3f}, N={self.num_particles} | Brain: t={brain_time_val:.2f} N={brain_neurons_val}, Syn={brain_synapses_val}, ThoughtR={brain_thought_region_val} | ZoneFactors Avg={np.mean(self.adaptive_factors_per_zone):.2f}")


        sim_end_time = time.time()
        print(f"--- Coupled Simulation Finished in {sim_end_time - sim_start_time:.2f} seconds ---")


    def run_analysis_and_visualization(self):
        """
        Runs analysis and visualization functions for the particle simulation results.
        Calls the corresponding functions with the stored histories.
        Brain simulation plotting is handled within the UnifiedBrainSimulation class.
        """
        print("--- Starting Analysis and Visualization ---")

        # Ensure history lists are not empty before analysis
        final_types_for_analysis = self.particle_types_history[-1] if self.particle_types_history else np.array([], dtype=int)


        rdf_bin_centers, rdf_g_r = calculate_radial_distribution_function(self.particle_positions_history, self.particle_parameters)
        if rdf_bin_centers.size > 0: plot_rdf(rdf_bin_centers, rdf_g_r, self.particle_parameters)

        if final_types_for_analysis.size > 0 or (self.particle_positions_history and len(self.particle_positions_history[0]) > 0) : # Only if types makes sense with particles or some particles existed
            msd_time_points, overall_msd, type_msd_dict = calculate_mean_squared_displacement(self.particle_positions_history, final_types_for_analysis, self.particle_parameters)
            if msd_time_points.size > 0 : plot_msd(msd_time_points, overall_msd, type_msd_dict, self.particle_parameters)


        bond_angle_bin_centers, bond_angle_counts = calculate_bond_angle_distribution(
            self.particle_positions_history, self.particle_orientations_history, self.particle_all_patch_data_history,
            self.particle_bonds_history, self.particle_types_history, self.particle_parameters
        )
        if bond_angle_bin_centers.size > 0 : plot_bond_angle_distribution(bond_angle_bin_centers, bond_angle_counts, self.particle_parameters)

        cluster_sizes, cluster_counts = calculate_cluster_size_distribution(self.particle_bonds_history, self.particle_positions_history, self.particle_parameters)
        if cluster_sizes.size > 0: plot_cluster_size_distribution(cluster_sizes, cluster_counts, self.particle_parameters)


        visualize_simulation(
            self.particle_total_energy_history, self.particle_positions_history, self.particle_orientations_history,
            self.particle_nematic_order_parameter_history, self.particle_all_patch_data_history,
            self.particle_bonds_history, self.particle_types_history, self.particle_parameters
        )

        self.brain_sim.plot_results() # Brain visualization


# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting script execution for interpretation...")
    parameters = load_parameters("coupled_parameters.json")

    coupled_sim = CoupledSimulation(parameters)
    coupled_sim.run_simulation()
    coupled_sim.run_analysis_and_visualization()
    print("Script execution for interpretation finished.")
