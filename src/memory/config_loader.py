import os
import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")

DEFAULT_MEMORY_CONFIG = {
    "enabled": False,
    "modules": {
        "episode_recorder": False,
        "component_evidence": False,
        "evolution_operators": False,
        "inter_component": False,
        "proposal_plan": False,
    },
    "evidence_settings": {
        "min_score_delta_for_effective": 2.0,
        "min_episodes_for_promote": 3,
        "maturity_threshold": 0.5,
        "freshness_lambda": 0.9,
        "effect_epsilon": 1e-6,
        "regression_threshold": -3.0,
        "small_validation_penalty": True,
    },
    "token_optimization": {
        "enabled": False,
        "plan_code_separation": True,
        "history_budget": True,
        "log_context_mode": True,
    },
}


def _load_yaml() -> dict:
    try:
        with open(os.path.abspath(_CONFIG_PATH)) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def _merge_memory_config(raw: dict | None) -> dict:
    cfg = {
        "enabled": DEFAULT_MEMORY_CONFIG["enabled"],
        "modules": dict(DEFAULT_MEMORY_CONFIG["modules"]),
        "evidence_settings": dict(DEFAULT_MEMORY_CONFIG["evidence_settings"]),
        "token_optimization": dict(DEFAULT_MEMORY_CONFIG["token_optimization"]),
        "optimizations": {},
        "warm_up": {},
    }
    if not isinstance(raw, dict):
        return cfg
    cfg["enabled"] = bool(raw.get("enabled", cfg["enabled"]))
    cfg["evidence_quality"] = bool(raw.get("evidence_quality", False))
    cfg["memory_efficiency"] = bool(raw.get("memory_efficiency", False))
    cfg["search_governance"] = bool(raw.get("search_governance", False))
    cfg["modules"].update(raw.get("modules") or {})
    cfg["evidence_settings"].update(raw.get("evidence_settings") or {})
    cfg["token_optimization"].update(raw.get("token_optimization") or {})
    cfg["optimizations"].update(raw.get("optimizations") or {})
    cfg["warm_up"].update(raw.get("warm_up") or {})
    return cfg


def get_memory_config() -> dict:
    return _merge_memory_config(_load_yaml().get("memory_config"))


def module_enabled_from_config(memory_config: dict, module_name: str) -> bool:
    return bool(
        memory_config.get("enabled", False)
        and memory_config.get("modules", {}).get(module_name, False)
    )


def module_enabled(module_name: str) -> bool:
    return module_enabled_from_config(get_memory_config(), module_name)


def memory_enabled() -> bool:
    """Global on/off switch."""
    return bool(get_memory_config().get("enabled", False))


def evidence_quality_enabled() -> bool:
    """True when the evidence-quality modules are active (component_evidence, evolution_operators, inter_component, direction_cluster, guidance_tracker)."""
    cfg = get_memory_config()
    return bool(cfg.get("enabled", False) and cfg.get("evidence_quality", False))


def memory_efficiency_enabled() -> bool:
    """True when the memory-efficiency modules are active (reading_strategy, component_playbook, token_optimization). Requires evidence_quality."""
    cfg = get_memory_config()
    return bool(
        cfg.get("enabled", False)
        and cfg.get("memory_efficiency", False)
        and evidence_quality_enabled()
    )


def search_governance_enabled() -> bool:
    """True when the search-governance modules are active (proposal_plan, constraint_layer)."""
    cfg = get_memory_config()
    return bool(cfg.get("enabled", False) and cfg.get("search_governance", False))


if __name__ == "__main__":
    cfg = get_memory_config()
    print(f"memory_config.enabled = {cfg.get('enabled')}")
    print()
    print("Category flags:")
    print(f"  memory_enabled:          {memory_enabled()}")
    print(f"  evidence_quality_enabled: {evidence_quality_enabled()}")
    print(f"  memory_efficiency_enabled: {memory_efficiency_enabled()}")
    print(f"  search_governance_enabled: {search_governance_enabled()}")
    print()
    print("Evidence settings:")
    for key, val in cfg.get("evidence_settings", {}).items():
        print(f"  {key}: {val}")
