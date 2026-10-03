"""Frozen experiment profiles and the model_CH codebook."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ModelCHSpec:
    """Documents what each numeric model_CH currently means in code."""

    model_id: int
    label: str
    defender_kind: str
    attacker_kind: str
    defender_reward_uses_p_expl: bool
    attacker_reward_uses_p_expl: bool
    defender_bellman: str
    attacker_forced_random: bool
    ch_ratio_feedback: bool
    notes: str = ""


MODEL_CH_CODEBOOK: Dict[int, ModelCHSpec] = {
    1: ModelCHSpec(
        1,
        "DQN defender + p_expl reward vs random attacker",
        "dqn",
        "random",
        True,
        False,
        "max_q",
        True,
        False,
    ),
    2: ModelCHSpec(
        2,
        "DQN defender + p_expl reward vs DQN attacker",
        "dqn",
        "dqn",
        True,
        False,
        "max_q",
        False,
        False,
    ),
    3: ModelCHSpec(
        3,
        "DQN defender + p_expl reward vs DQN attacker + p_expl",
        "dqn",
        "dqn",
        True,
        True,
        "max_q",
        False,
        False,
    ),
    4: ModelCHSpec(
        4,
        "CHT-DQN defender vs random attacker",
        "cht_dqn",
        "random",
        False,
        False,
        "cht_scaled_max_q",
        True,
        True,
        "CH ratio feedback is active in main_ch1* / transition-aware paths.",
    ),
    5: ModelCHSpec(
        5,
        "CHT-DQN defender vs DQN attacker",
        "cht_dqn",
        "dqn",
        False,
        False,
        "cht_scaled_max_q",
        False,
        True,
        "Canonical HITL Scenario 3 / transition-aware matched control.",
    ),
    6: ModelCHSpec(
        6,
        "CHT-DQN defender vs DQN attacker + p_expl",
        "cht_dqn",
        "dqn",
        False,
        True,
        "cht_scaled_max_q",
        False,
        True,
    ),
    7: ModelCHSpec(
        7,
        "DQN defender vs random attacker",
        "dqn",
        "random",
        False,
        False,
        "max_q",
        True,
        False,
    ),
    8: ModelCHSpec(
        8,
        "DQN defender vs DQN attacker",
        "dqn",
        "dqn",
        False,
        False,
        "max_q",
        False,
        False,
        "Canonical HITL Scenario 4 / reward-aware matched control.",
    ),
    9: ModelCHSpec(
        9,
        "DQN defender vs DQN attacker + p_expl",
        "dqn",
        "dqn",
        False,
        True,
        "max_q",
        False,
        False,
    ),
}


@dataclass(frozen=True)
class ExperimentProfile:
    name: str
    kernel: str
    model_ch: int
    node_max: int
    time_max: int
    batch_size: int
    buffer_size: int
    update_interval: int
    gamma: float
    learning_rate: float
    betta: float
    eps_start: float
    eps_end: float
    eps_decay: float
    eps_schedule: str
    target_update_every: int
    target_update_mode: str
    softmax_epsilon: float
    result_scenario: str


PROFILES: Dict[str, ExperimentProfile] = {
    "offline_long_run_dqn": ExperimentProfile(
        name="offline_long_run_dqn",
        kernel="offline_v1",
        model_ch=8,
        node_max=6,
        time_max=2000,
        batch_size=64,
        buffer_size=1_000_000,
        update_interval=100,
        gamma=0.98,
        learning_rate=5e-2,
        betta=1.0,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=2.0,
        eps_schedule="linear_offline",
        target_update_every=100,
        target_update_mode="every_n_env_steps",
        softmax_epsilon=0.0,
        result_scenario="def DQN_att DQN",
    ),
    "offline_long_run_cht": ExperimentProfile(
        name="offline_long_run_cht",
        kernel="offline_v1",
        model_ch=5,
        node_max=6,
        time_max=2000,
        batch_size=64,
        buffer_size=1_000_000,
        update_interval=100,
        gamma=0.98,
        learning_rate=5e-2,
        betta=1.0,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=2.0,
        eps_schedule="linear_offline",
        target_update_every=100,
        target_update_mode="every_n_env_steps",
        softmax_epsilon=0.0,
        result_scenario="DQN-CH-level1 def DQN p_expl_att DQN",
    ),
    "matched_dqn": ExperimentProfile(
        name="matched_dqn",
        kernel="matched_v2",
        model_ch=8,
        node_max=6,
        time_max=40,
        batch_size=5,
        buffer_size=1_000,
        update_interval=5,
        gamma=0.98,
        learning_rate=5e-2,
        betta=1.0,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=0.93,
        eps_schedule="exponential",
        target_update_every=1,
        target_update_mode="every_learn",
        softmax_epsilon=0.0,
        result_scenario="def DQN_att DQN",
    ),
    "matched_cht": ExperimentProfile(
        name="matched_cht",
        kernel="matched_v2",
        model_ch=5,
        node_max=6,
        time_max=40,
        batch_size=5,
        buffer_size=1_000,
        update_interval=5,
        gamma=0.98,
        learning_rate=5e-2,
        betta=1.0,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=0.93,
        eps_schedule="exponential",
        target_update_every=1,
        target_update_mode="every_learn",
        softmax_epsilon=0.0,
        result_scenario="DQN-CH-level1 def DQN p_expl_att DQN",
    ),
    "hitl_reward_aware": ExperimentProfile(
        name="hitl_reward_aware",
        kernel="web_v3",
        model_ch=8,
        node_max=6,
        time_max=40,
        batch_size=5,
        buffer_size=10_000,
        update_interval=5,
        gamma=0.98,
        learning_rate=5e-2,
        betta=1.0,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=0.93,
        eps_schedule="exponential",
        target_update_every=1,
        target_update_mode="every_learn",
        softmax_epsilon=1e-50,
        result_scenario="reward_aware",
    ),
    "hitl_transition_aware": ExperimentProfile(
        name="hitl_transition_aware",
        kernel="web_v3",
        model_ch=5,
        node_max=6,
        time_max=40,
        batch_size=5,
        buffer_size=10_000,
        update_interval=5,
        gamma=0.98,
        learning_rate=5e-2,
        betta=1.0,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=0.93,
        eps_schedule="exponential",
        target_update_every=1,
        target_update_mode="every_learn",
        softmax_epsilon=1e-50,
        result_scenario="transition_aware",
    ),
}


def profile_dict(name: str) -> Dict:
    return asdict(PROFILES[name])


def codebook_rows() -> List[Dict]:
    return [asdict(spec) for spec in MODEL_CH_CODEBOOK.values()]
