"""Local path and session-secret configuration for both web conditions."""

import os
import secrets
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = Path(
    os.environ.get("CHTDQN_RUNTIME_DIR", PROJECT_ROOT / ".runtime")
).resolve()

DATA_ENV_NAMES = {
    "reward_aware": "CHTDQN_REWARD_AWARE_DATA_DIR",
    "transition_aware": "CHTDQN_TRANSITION_AWARE_DATA_DIR",
}


def condition_paths(condition):
    """Create and return data, pickle, session, and log paths."""
    data_dir = Path(
        os.environ.get(
            DATA_ENV_NAMES[condition],
            PROJECT_ROOT / "restricted_data" / "hitl" / "raw" / condition,
        )
    ).resolve()
    pickle_dir = RUNTIME_ROOT / "pickles" / condition
    session_dir = RUNTIME_ROOT / "sessions" / condition
    log_file = RUNTIME_ROOT / "logs" / f"{condition}.log"

    for directory in (data_dir, pickle_dir, session_dir, log_file.parent):
        directory.mkdir(parents=True, exist_ok=True)

    return data_dir, pickle_dir, session_dir, log_file


def flask_secret_key():
    """Return an environment secret or a stable local-only generated secret."""
    configured = os.environ.get("FLASK_SECRET_KEY")
    if configured:
        return configured

    secret_file = RUNTIME_ROOT / "flask_secret"
    secret_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        return secret_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        generated = secrets.token_hex(32)
        try:
            with secret_file.open("x", encoding="utf-8") as file:
                file.write(generated)
            secret_file.chmod(0o600)
            return generated
        except FileExistsError:
            return secret_file.read_text(encoding="utf-8").strip()
