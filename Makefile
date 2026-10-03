PYTHON ?= python3
GUNICORN ?= gunicorn
export PYTHONPATH := src:$(PYTHONPATH)

.PHONY: offline cht-offline matched-dqn matched-cht matched-sweep matched-plots web-reward web-transition analyze stats test

offline:
	cd components/offline_simulation && $(PYTHON) main.py

cht-offline:
	cd components/offline_simulation && $(PYTHON) main_ch1_salient.py

matched-dqn:
	cd components/web_matched_simulations && $(PYTHON) main.py

matched-cht:
	cd components/web_matched_simulations && $(PYTHON) main_ch1.py

matched-sweep:
	$(PYTHON) components/web_matched_simulations/batch/run_matched_controls.py

matched-plots:
	$(PYTHON) components/web_matched_simulations/analysis/plot_convergence.py

web-reward:
	cd components/web_experiment && $(GUNICORN) app_reward_aware:app -b 0.0.0.0:5001

web-transition:
	cd components/web_experiment && $(GUNICORN) app_transition_aware:app -b 0.0.0.0:5002

analyze:
	$(PYTHON) analysis/hitl/protection_and_time.py
	$(PYTHON) analysis/hitl/action_frequencies.py
	$(PYTHON) analysis/hitl/prospect_behavior.py

stats:
	$(PYTHON) analysis/stats/run_all.py

test:
	$(PYTHON) -m unittest discover -s tests -v
