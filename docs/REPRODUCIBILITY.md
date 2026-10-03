# Reproducibility

## Environment limitation

The source projects did not include pinned Python environments. `requirements/` records inferred direct dependencies but cannot reproduce the exact historical environment. Record tested package versions after establishing a working environment.

Historical bytecode indicates that Python 3.6 and 3.8 were used at different times. Start with a current supported Python release, but expect compatibility work for old PyTorch/TensorFlow-era code.

Unused TensorFlow imports were removed from the cleaned working copy; the models are implemented in PyTorch.

## Working directories

Run each simulation from its own component directory. Their imports and result paths are local to that directory.

```bash
cd components/offline_simulation
python main.py
```

```bash
cd components/web_matched_simulations
python main.py
```

Run HITL analysis from the repository root:

```bash
python analysis/hitl/protection_and_time.py
```

## Known limitations preserved from the research code

- Broad offline defaults contain a float floor-division issue in evaluation slicing.
- Some random-number seeding is incomplete in the web-matched simulations.
- The matched target networks update every learning step, unlike the broad simulation’s 100-step lag.
- Some command-line vectors are overwritten by defaults later in execution.
- Conceptual figures (Figure 5) and interface screenshots (Figure 1) were not generated programmatically.

## Paper-output lineage

- Figure 1: screenshots of the two web games (`components/web_experiment/`).
- Figure 2: `analysis/hitl/protection_and_time.py`.
- Figure 3: `analysis/hitl/prospect_behavior.py`.
- Figure 4: `analysis/hitl/action_frequencies.py`.
- Table 1: `analysis/stats/02_hitl_primary_outcomes.py`, `03_hitl_mixed_effects.py`, `04_hitl_prospect_tests.py`.
- Figure 6: `analysis/stats/05_plot_offline_protection.py`.
- Figure 7: `components/offline_simulation/batch/run_reward_curves.py`.
