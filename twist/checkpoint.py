"""Save training state while leaving runtime hyperparameters in current config."""
from __future__ import annotations


def optimizer_state(optimizer):
    """Keep Adam moments and parameter IDs, without saved lr/betas/options."""
    return optimizer_moments(optimizer.state_dict())


def optimizer_moments(state):
    return {"state": state["state"],
            "param_groups": [{"params": group["params"]} for group in state["param_groups"]]}


def restore_optimizer(optimizer, saved):
    """Restore moments using the options of the freshly created optimizer.

    Legacy snapshots containing optimizer options are supported; those options
    do not override this invocation's configuration.
    """
    current = optimizer.state_dict()["param_groups"]
    if len(current) != len(saved["param_groups"]):
        raise ValueError("optimizer parameter groups differ from checkpoint")
    groups = [{**group, "params": old["params"]}
              for group, old in zip(current, saved["param_groups"])]
    optimizer.load_state_dict({"state": saved["state"], "param_groups": groups})


def state_only(snapshot):
    """Remove runtime settings, including from older snapshots being rescored."""
    result = {key: value for key, value in snapshot.items()
              if key not in ("training_config", "loss_config")}
    result["optimizer"] = optimizer_moments(result["optimizer"])
    if "metrics" in result:
        result["metrics"] = {key: value for key, value in result["metrics"].items()
                             if key != "loss_config"}
    return result
