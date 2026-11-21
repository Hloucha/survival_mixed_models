# survival_mixed_models

Code for comparing right-censored survival data across samples or treatments when strong run-to-run variability is expected.

## Experimental Structure
- **Samples / treatments** ($n$)
- **Repeated runs** ($n_{\text{runs}}$) per sample/treatment
- **Observations** ($n_{\text{obs}}$) per run (including right-censored data)

## Approach
- Survival mixed-effects model with **treatment/sample** as a fixed effect and **run** as a random effect.
