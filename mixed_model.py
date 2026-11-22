"""
Code for comparing right-censored survival data across samples or treatments when strong 
run-to-run variability is expected.

Experimental Structure
Samples: n_samples
Repeated runs: n_runs (they contain only one sample type)
Observations for each run: n_obs per run 


First part:
generation of simulated data


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
from statsmodels.duration.survfunc import SurvfuncRight
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf


#%%
# B) functions


def show_info():
    """Prints environment information including the current user, working directory """
    print(f'user: {getpass.getuser()}')
    print(f'path: {os.getcwd()}')





def generate_plate_dataset_cox(
    n_samples: int = 5,
    n_runs: int = 5,
    n_obs: int = 15,
    run_sd: float = 0.2,
    shape: float = 1.0, # Weibull shape: 1 means constant hazard over time.
    lambda_0: float = 0.02,
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

    rng = np.random.default_rng(seed) #self contained generator, faster, better stats

    df = pd.DataFrame()
    # log HR for samples, A is set to 0, the rest random  between log(0.5) and log(2)
    sample_effects = np.zeros(n_samples)
    sample_effects[1:] = rng.uniform(np.log(0.5), np.log(2.0), size=n_samples - 1)
    

    # loop over samples
    for sample in range(n_samples):
        sample_name = string.ascii_uppercase[sample] # make sample name A-Z
        sample_effect = sample_effects[sample]  # log HR for this sample

        # loop over runs for this sample
        for run in range(n_runs):
            run_effect = rng.normal(0.0, run_sd)
            
            # eta: total linear predictor (log-HR) for that specific (sample, run)
            eta = sample_effect + run_effect 

            #  scale is the characteristic survival time (1/lambda) of the Weibull distribution 
            scale = (lambda_0 * np.exp(eta)) ** (-1.0 / shape)

            # loop over observations for this run
            for obs in range(n_obs):

                #  rng.weibull(a=1.0) * scale is identical to rng.exponential(scale=scale)
                raw_time = rng.weibull(a=shape) * scale
                if raw_time <= t_max:
                    event_time = raw_time
                    obs_event = 1
                else:
                    event_time = t_max
                    obs_event = 0

                df_i = pd.DataFrame([{'sample': sample_name, 'run': f'{sample_name}_{run}', 'obs': obs, 'event_time': event_time, 'obs_event': obs_event}])
                df = pd.concat([df, df_i], ignore_index=True)
            

    return df


def plot_survival_by_sample(df: pd.DataFrame) -> None:
    """Plot Kaplan-Meier survival curves per sample, with one line/color per run.

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset with columns: sample, run, event_time, obs_event.
    """
    samples = sorted(df['sample'].unique())
    n_samples = len(samples)
    max_time = df['event_time'].max()
    # Okabe-Ito colorblind-friendly palette
    palette = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00', '#56B4E9', '#F0E442', '#000000']

    fig, axes = plt.subplots(
        1, n_samples, figsize=(3.8 * n_samples, 3.8), sharex=True, sharey=True, squeeze=False
    )

    for ax, sample in zip(axes.ravel(), samples):
        df_s = df[df['sample'] == sample]
        for idx, (run_id, df_r) in enumerate(df_s.groupby('run', sort=False)):
            sf = SurvfuncRight(df_r['event_time'].to_numpy(), df_r['obs_event'].to_numpy())
            times = np.concatenate([[0.0], sf.surv_times, [max_time]])
            probs = np.concatenate([[1.0], sf.surv_prob, [sf.surv_prob[-1] if len(sf.surv_prob) else 1.0]])
            ax.step(times, probs, where='post', color=palette[idx % len(palette)], label=str(run_id))

        ax.set_title(f'Sample {sample}')
        ax.set_xlabel('Time (a.u.)')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(title='Run', fontsize=8, frameon=False)

    axes[0, 0].set_ylabel('Survival probability')
    fig.tight_layout()
    plt.show()


df_sim = generate_plate_dataset_cox()
plot_survival_by_sample(df_sim)



   
# %%
