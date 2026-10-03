# Human Web Experiment

Two Flask/D3 applications implement the human-in-the-loop conditions:

- `app_reward_aware.py` — reward-aware interface, historically served on port 5001
- `app_transition_aware.py` — reward + transition-aware interface, historically served on port 5002

Run from this directory so the component-local `agents`, `buffer`, `envs`, `templates`, and `static` paths resolve correctly.

```bash
export FLASK_SECRET_KEY="a-stable-secret-shared-by-all-workers"
gunicorn app_reward_aware:app -b 0.0.0.0:5001
gunicorn app_transition_aware:app -b 0.0.0.0:5002
```

Default local outputs:

- participant JSON: `../../restricted_data/hitl/raw/`
- logs, Flask sessions, and pickle state: `../../.runtime/`

The locations can be overridden with the environment variables documented in the repository’s `.env.example`.

The historical EC2 address, absolute server paths, private key, runtime caches, and logs are not part of this cleaned component.
