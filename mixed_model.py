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
from scipy import stats
from statsmodels.duration.survfunc import SurvfuncRight
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf


#%%
# B) functions


def show_info():
    """Prints environment information including the current user, working directory """
    print(f'user: {getpass.getuser()}')
    print(f'path: {os.getcwd()}')



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


def median_estimate(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Estimate per-sample median survival time and t-based CI from run-level KM medians.

    Computes a Kaplan-Meier median for each run, checks that all runs have a
    defined median, and aggregates on the log scale using a t-interval across runs.

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset with columns: sample, run, event_time, obs_event.
    alpha : float, optional
        Significance level for the two-sided confidence interval (default 0.05).

    Returns
    -------
    pd.DataFrame
        Columns: sample, n_runs, median, ci_low, ci_high.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1")

    run_rows = []
    for (sample, run_id), df_r in df.groupby(['sample', 'run'], sort=False):
        sf = SurvfuncRight(df_r['event_time'].to_numpy(), df_r['obs_event'].to_numpy())
        run_rows.append({'sample': sample, 'run': run_id, 'median': float(sf.quantile(0.5))})

    run_df = pd.DataFrame(run_rows)
    missing = run_df[run_df['median'].isna() | (run_df['median'] <= 0)] # no median 
    if not missing.empty:
        missing_runs = [f"{r.sample}/{r.run}" for r in missing.itertuples(index=False)]
        raise ValueError(
            f"KM median undefined (censoring >= 50%) for runs: {', '.join(missing_runs)}"
        )

    results = []
    for sample, grp in run_df.groupby('sample', sort=True):
        n_runs = len(grp)
        if n_runs < 2:
            raise ValueError(f"Sample '{sample}' has only {n_runs} run(s); need >= 2 for t-CI")

        log_medians = np.log(grp['median'].to_numpy())
        mean_log = float(np.mean(log_medians))
        se_log = float(np.std(log_medians, ddof=1) / np.sqrt(n_runs))
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n_runs - 1))

        results.append({
            'sample': sample,
            'n_runs': n_runs,
            'median': float(np.exp(mean_log)),
            'ci_low': float(np.exp(mean_log - t_crit * se_log)),
            'ci_high': float(np.exp(mean_log + t_crit * se_log)),
        })

    return pd.DataFrame(results)


def generate_plate_dataset_cox(
    n_samples: int = 5,
    n_runs: int = 5,
    n_obs: int = 15,
    run_sd: float = 0.2,
    shape: float = 1.0,  # Weibull shape: 1 means constant hazard over time.
    lambda_0: float = 0.02,
    t_max: float = 100.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Simulate right-censored survival data for a Cox mixed-effects model.

    Parameters
    ----------
    n_samples : int
        Number of samples/treatments (1 to 26, named A-Z).
    n_runs : int
        Number of independent runs/plates per sample.
    n_obs : int
        Number of observations per sample per run.
    run_sd : float
        Standard deviation of the Gaussian run random effect (log-HR scale).
    shape : float
        Weibull shape parameter (> 0).
    lambda_0 : float
        Baseline hazard scale parameter (> 0).
    t_max : float
        Right-censoring cutoff time (> 0).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    df : pd.DataFrame
        Columns: sample, run, obs, event_time, obs_event.
    true_effects : dict[str, float]
        Ground-truth log-hazard ratios keyed by sample name (e.g., {'A': 0.0, ...}).
    """
    if not (1 <= n_samples <= 26):
        raise ValueError("n_samples must be between 1 and 26 (names A-Z)")
    if n_runs < 1 or n_obs < 1:
        raise ValueError("n_runs and n_obs must be at least 1")
    if run_sd < 0:
        raise ValueError("run_sd must be non-negative")
    if shape <= 0 or lambda_0 <= 0 or t_max <= 0:
        raise ValueError("shape, lambda_0, and t_max must be positive")

    rng = np.random.default_rng(seed)  # self-contained generator, faster, better stats

    df = pd.DataFrame()
    # log HR for samples: A is reference (0.0), others random between log(0.5) and log(2.0)
    sample_effects = np.zeros(n_samples)
    sample_effects[1:] = rng.uniform(np.log(0.5), np.log(2.0), size=n_samples - 1)
    true_effects = {}

    # loop over samples
    for sample in range(n_samples):
        sample_name = string.ascii_uppercase[sample]  # make sample name A-Z
        sample_effect = sample_effects[sample]  # log HR for this sample
        true_effects[sample_name] = float(sample_effect)

        # loop over runs for this sample
        for run in range(n_runs):
            run_effect = rng.normal(0.0, run_sd)

            # eta: total linear predictor (log-HR) for that specific (sample, run)
            eta = sample_effect + run_effect

            # scale is the characteristic survival time (1/lambda) of the Weibull distribution
            scale = (lambda_0 * np.exp(eta)) ** (-1.0 / shape)

            # loop over observations for this run
            for obs in range(n_obs):
                # rng.weibull(a=1.0) * scale is identical to rng.exponential(scale=scale)
                raw_time = rng.weibull(a=shape) * scale
                if raw_time <= t_max:
                    event_time = raw_time
                    obs_event = 1
                else:
                    event_time = t_max
                    obs_event = 0

                df_i = pd.DataFrame([{
                    'sample': sample_name,
                    'run': f'{sample_name}_{run}',
                    'obs': obs,
                    'event_time': event_time,
                    'obs_event': obs_event,
                }])
                df = pd.concat([df, df_i], ignore_index=True)

    return df, true_effects

#%%
# C) RUN

df_sim, true_eff = generate_plate_dataset_cox()

plot_survival_by_sample(df_sim)
print(median_estimate(df_sim))

   
# %%
