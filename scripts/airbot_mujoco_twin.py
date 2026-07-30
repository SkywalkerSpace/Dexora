"""Airbot MuJoCo digital-twin replay adapter.

This module is intentionally independent from the training stack.  It can be
used by ``scripts/replay_validate.py`` through the small ``replay`` API.

Backends
--------
``AIRBOT_MUJOCO_TASK_MODULE`` (recommended)
    A Discoverse task module containing ``SimNode`` and ``cfg``.  The adapter
    drives the task's ``step(action)`` method.

``AIRBOT_MUJOCO_XML``
    A plain MuJoCo XML.  This is useful for a model/sensor smoke test.  The
    first ``min(nq, nu, action_dim)`` controls are interpreted as joint
    position targets and held with a stable PD controller.

The XML backend deliberately reports task success as false unless a task
success callback is supplied.  A physically stable replay is not equivalent
to completing the manipulation task.
"""
from __future__ import annotations

import importlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _env_flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(int(default))).lower() in {"1", "true", "yes", "on"}


@dataclass
class _Result:
    success: bool
    collision_free: bool
    steps: int
    note: str = ""
    failure_reason: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {k: v for k, v in self.__dict__.items() if v not in ("", None)}


class _XmlTwin:
    def __init__(self, xml_path: str):
        try:
            import mujoco  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional env
            raise RuntimeError("Install MuJoCo first: pip install mujoco") from exc

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(Path(xml_path).expanduser()))
        self.data = mujoco.MjData(self.model)
        self.control_dim = min(self.model.nu, self.model.nq)
        self.button_joint = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_JOINT, "button_slide"
        )
        self.button_min = 0.012
        self.button_pressed = False
        self.viewer = None
        if _env_flag("AIRBOT_MUJOCO_GUI", False):
            try:
                import mujoco.viewer  # type: ignore
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except Exception as exc:
                raise RuntimeError(
                    "MuJoCo GUI could not start; run with AIRBOT_MUJOCO_GUI=0 on headless machines"
                ) from exc
        if self.control_dim == 0:
            raise ValueError("MuJoCo XML has no actuators or joint coordinates")

    def reset(self, states: np.ndarray) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        if states.ndim == 2 and states.shape[0] and states.shape[1]:
            n = min(self.model.nq, states.shape[1])
            self.data.qpos[:n] = states[0, :n]
        self.mujoco.mj_forward(self.model, self.data)
        self.button_pressed = False

    def step(self, action: np.ndarray) -> bool:
        n = min(self.control_dim, action.size)
        self.data.ctrl[:n] = np.asarray(action[:n], dtype=float)
        self.mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()
            time.sleep(max(0.0, float(self.model.opt.timestep)))
        return bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())

    def collision_free(self) -> bool:
        # The click task allows pen-button contact. Other contacts are treated
        # as collisions; task XMLs can provide a task-specific callback when
        # they need richer contact filtering.
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            n1 = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, g1) or ""
            n2 = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, g2) or ""
            if not ({n1, n2} <= {"pen_tip", "button"}):
                return False
        return True

    def success(self) -> bool:
        if self.button_joint < 0:
            return False
        qadr = int(self.model.jnt_qposadr[self.button_joint])
        pressed = float(self.data.qpos[qadr]) >= self.button_min
        self.button_pressed = self.button_pressed or pressed
        return bool(self.button_pressed)


class _DiscoverseTwin:
    def __init__(self, module_name: str):
        module = importlib.import_module(module_name)
        if not hasattr(module, "SimNode") or not hasattr(module, "cfg"):
            raise ImportError(f"{module_name} must expose SimNode and cfg")
        cfg = module.cfg
        if hasattr(cfg, "headless"):
            cfg.headless = not _env_flag("AIRBOT_MUJOCO_GUI", False)
        self.task_success = getattr(module, "task_success", None)
        self.task_collision_free = getattr(module, "collision_free", None)
        self.node = module.SimNode(cfg)

    def reset(self, states: np.ndarray) -> None:
        self.node.reset()
        # Discoverse task implementations differ in how they expose qpos.
        # If available, initialize from the recorded state and forward once.
        if states.ndim == 2 and states.shape[0] and states.shape[1]:
            data = getattr(self.node, "mj_data", None)
            if data is not None and hasattr(data, "qpos"):
                n = min(len(data.qpos), states.shape[1])
                data.qpos[:n] = states[0, :n]
                model = getattr(self.node, "mj_model", None)
                if model is not None:
                    try:
                        import mujoco  # type: ignore
                        mujoco.mj_forward(model, data)
                    except ImportError:
                        pass

    def step(self, action: np.ndarray) -> bool:
        self.node.step(np.asarray(action, dtype=float))
        data = getattr(self.node, "mj_data", None)
        return data is None or bool(np.isfinite(data.qpos).all())

    def collision_free(self) -> bool:
        if callable(self.task_collision_free):
            return bool(self.task_collision_free(self.node))
        data = getattr(self.node, "mj_data", None)
        if data is None:
            return True
        # Discoverse exposes MuJoCo's contact array in the usual format.  The
        # task-specific success/collision policy can override this module.
        return int(getattr(data, "ncon", 0)) == 0

    def success(self) -> Optional[bool]:
        if callable(self.task_success):
            return bool(self.task_success(self.node))
        return None


def _make_twin() -> Any:
    task = os.getenv("AIRBOT_MUJOCO_TASK_MODULE")
    xml = os.getenv("AIRBOT_MUJOCO_XML")
    if task:
        return _DiscoverseTwin(task)
    if xml:
        return _XmlTwin(xml)
    raise RuntimeError(
        "Set AIRBOT_MUJOCO_TASK_MODULE (Discoverse task) or AIRBOT_MUJOCO_XML (MuJoCo XML)"
    )


def replay(states: np.ndarray, actions: np.ndarray, task_id: int) -> Dict[str, object]:
    """Replay one episode and return the verifier result expected by Dexora."""
    del task_id
    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=float)
    if actions.ndim != 2 or actions.shape[0] == 0:
        return _Result(False, False, 0, failure_reason="empty or non-matrix actions").as_dict()
    if not np.isfinite(actions).all():
        return _Result(False, False, 0, failure_reason="actions contain NaN/Inf").as_dict()

    twin = _make_twin()
    twin.reset(states)
    for i, action in enumerate(actions):
        try:
            if not twin.step(action):
                return _Result(False, False, i + 1, failure_reason="non-finite MuJoCo state").as_dict()
        except Exception as exc:
            return _Result(False, False, i + 1, failure_reason=f"step error: {exc}").as_dict()

    collision_free = bool(twin.collision_free())
    # A success callback may be provided by a task module.  Without one, XML
    # replay is only a physics smoke test and must not enter S_high as success.
    task_success = twin.success() if hasattr(twin, "success") else None
    success = _env_flag("AIRBOT_MUJOCO_ASSUME_SUCCESS", False) if task_success is None else task_success
    return _Result(
        success,
        collision_free,
        len(actions),
        note="Airbot MuJoCo replay; success requires task callback or explicit assume-success",
        failure_reason="task success predicate not configured" if not success else "",
    ).as_dict()


def _run_gui(xml_path: str, steps: int) -> None:
    """Play a stationary/zero-action XML scene and keep the window open."""
    os.environ["AIRBOT_MUJOCO_XML"] = xml_path
    os.environ["AIRBOT_MUJOCO_GUI"] = "1"
    twin = _XmlTwin(xml_path)
    actions = np.zeros((steps, 6), dtype=float)
    twin.reset(np.zeros((1, 1)))
    for action in actions:
        twin.step(action)
    print({"success": twin.success(), "collision_free": twin.collision_free(), "steps": steps})
    print("MuJoCo viewer is open; close the window to exit.")
    while twin.viewer is not None and twin.viewer.is_running():
        time.sleep(0.1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test an Airbot MuJoCo XML")
    parser.add_argument("--xml", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--gui", action="store_true", help="open the interactive MuJoCo viewer")
    args = parser.parse_args()
    if args.gui:
        _run_gui(args.xml, args.steps)
    else:
        os.environ["AIRBOT_MUJOCO_XML"] = args.xml
        print(replay(np.zeros((1, 1)), np.zeros((args.steps, 6)), 0))
