import sys


def _sync_package_aliases() -> None:
    """Keep `memory.*` and `src.memory.*` as one module tree."""
    current = sys.modules[__name__]
    if __name__ == "src.memory":
        sys.modules["memory"] = current
    elif __name__ == "memory":
        sys.modules["src.memory"] = current

    for name, module in list(sys.modules.items()):
        if name.startswith("src.memory."):
            sys.modules[name.replace("src.", "", 1)] = module
        elif name.startswith("memory."):
            sys.modules[f"src.{name}"] = module


def _alias_submodule(short_name: str, module) -> None:
    sys.modules[f"src.memory.{short_name}"] = module
    sys.modules[f"memory.{short_name}"] = module


_sync_package_aliases()


def configure(task: str) -> None:
    """Configure memory for 'text' (classification) or 'terminal' (Terminal-Bench/SWE-bench).

    Call once at process start before any memory functions are used.
    """
    from src.memory.encoding import component_evidence
    from src.memory.steering import component_playbook
    from src.memory.encoding import episode_recorder

    _alias_submodule("component_evidence", component_evidence)
    _alias_submodule("component_playbook", component_playbook)
    _alias_submodule("episode_recorder", episode_recorder)

    component_evidence.configure(task)
    component_playbook.configure(task)
    episode_recorder.configure(task)

    # If modules with cached component references are already loaded, point them
    # at the configured component_evidence module.
    for module_name in (
        "src.memory.steering.reading_strategy",
        "memory.steering.reading_strategy",
    ):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "_ce"):
            module._ce = component_evidence

    _sync_package_aliases()
