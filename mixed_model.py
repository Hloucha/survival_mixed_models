
"""
Code for comparing right-censored survival data across samples or treatments when strong 
run-to-run variability is expected.

Experimental Structure
Samples: n_samples
Repeated runs: n_runs
Observations for each run: n_obs per run 


env: UV
MH, November 2026

"""

#%%
# A) libs and path
import pandas as pd
import numpy as np
import os
import string
import getpass
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf


#%%
# B) functions


def show_info():
    """Prints environment information including the current user, working directory """
    print(f'user: {getpass.getuser()}')
    print(f'path: {os.getcwd()}')



def generate_plate_dataset (n_samples, n_runs, n_obs):
    np.random.seed(42)

    t_max = 100 # max observation time
    range_surv_min = 10 # range for mean survival times
    range_surv_max = 50

    run_shift_max = 5 # run shift
    obs_shift_max = 2 # observation variance


    df = pd.DataFrame()
    if n_samples > 26:
        raise ValueError("too many samples for names A-Z")
    else:
        for sample in range(n_samples):
            sample_name = string.ascii_uppercase[sample]
            sample_mean = np.random.uniform(range_surv_min, range_surv_max)
            for run in range(n_runs):
                run_shift = np.random.uniform(-run_shift_max, run_shift_max)
                for obs in range(n_obs):
                    obs_shift = np.random.uniform(-obs_shift_max, obs_shift_max)
                    event_time = sample_mean + run_shift + obs_shift
                    if event_time <= t_max:
                        obs_event = 1
                    else:
                        obs_event = 0
                    df_i = pd.DataFrame({'sample': [sample_name], 'run': [run], 'obs': [obs], 'event_time': [event_time], 'obs_event': [obs_event]})
                    df = pd.concat([df, df_i])
    return df


def generate_plate_dataset_cox(
    n_samples: int = 5,
    n_runs: int = 5,
    n_obs: int = 15,
    run_sd: float = 0.8,
    shape: float = 1.5,
    lambda_0: float = 0.001,
    t_max: float = 100.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate right-censored survival data for a Cox mixed-effects model.

    Parameters
    ----------
    n_samples : int
        Number of samples/treatments (max 26, named A-Z).
    n_runs : int
        Number of independent runs/plates.
    n_obs : int
        Number of observations per sample per run.
    run_sd : float
        Standard deviation of the Gaussian run random effect (log-HR scale).
    shape : float
        Weibull shape parameter (> 0).
    lambda_0 : float
        Baseline hazard scale parameter (> 0).
    t_max : float
        Right-censoring cutoff time.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: sample, run, obs, event_time, obs_event.
    """
    if n_samples > 26:
        raise ValueError("too many samples for names A-Z")

    rng = np.random.default_rng(seed)
    sample_effects = np.linspace(0.0, 1.0, n_samples)  # true log-HR vs sample A
    rows = []

    for run in range(n_runs):
        run_effect = rng.normal(0.0, run_sd)
        for sample in range(n_samples):
            sample_name = string.ascii_uppercase[sample]
            
            eta = sample_effects[sample] + run_effect
            scale = (lambda_0 * np.exp(eta)) ** (-1.0 / shape)
            raw_times = rng.weibull(a=shape, size=n_obs) * scale
            obs_times = np.minimum(raw_times, t_max)
            obs_events = (raw_times <= t_max).astype(int)

            for obs, (t, e) in enumerate(zip(obs_times, obs_events)):
                rows.append({
                    'sample': sample_name,
                    'run': run,
                    'obs': obs,
                    'event_time': t,
                    'obs_event': e,
                })

    return pd.DataFrame(rows)


generate_plate_dataset (n_samples = 5, n_runs = 5, n_obs = 15)



   
# %%
