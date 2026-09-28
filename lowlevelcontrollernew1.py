import numpy as np
import torch

from ray.rllib.algorithms.sac import SAC
from ray.tune.registry import register_env

from rsacenv import UAVRSACEnv


# =========================================================
# REGISTER ENVIRONMENT REQUIRED BY CHECKPOINT
# =========================================================

def env_creator(config):
    return UAVRSACEnv()


register_env(
    "low_level_env",
    env_creator
)


class LowLevelController:

    def __init__(self, agents):

        self.agents = agents

        # =====================================================
        # LOAD TRAINED RECURRENT SAC CHECKPOINT
        # =====================================================
        self.algo = SAC.from_checkpoint(
            "E:/uav_hmarl_project/rsac_lstm_test_checkpoint"
        )

        # =====================================================
        # GET RL MODULE
        # =====================================================
        self.module = self.algo.get_module()

        self.action_dist_cls = (
            self.module.get_inference_action_dist_cls()
        )

        # =====================================================
        # INITIAL RECURRENT STATES
        # =====================================================
        self.hidden_states = {}

        for agent in agents:

            initial_state = self.module.get_initial_state()

            self.hidden_states[agent] = self._copy_state(
                initial_state
            )

            print(
                f"[LOW INIT] {agent} | "
                f"initial_state="
                f"{self._state_shapes(initial_state)}"
            )

        print(
            "\n[LOW] Recurrent SAC RLModule loaded successfully"
        )

    # =========================================================
    # COPY STATE
    # =========================================================
    def _copy_state(self, state):

        if state is None:
            return None

        if isinstance(state, dict):
            return {
                k: self._copy_state(v)
                for k, v in state.items()
            }

        if isinstance(state, list):
            return [
                self._copy_state(v)
                for v in state
            ]

        if isinstance(state, tuple):
            return tuple(
                self._copy_state(v)
                for v in state
            )

        if torch.is_tensor(state):
            return state.clone()

        return state

    # =========================================================
    # STATE SHAPE DEBUG
    # =========================================================
    def _state_shapes(self, state):

        if state is None:
            return None

        if isinstance(state, dict):
            return {
                k: self._state_shapes(v)
                for k, v in state.items()
            }

        if isinstance(state, (list, tuple)):
            return [
                self._state_shapes(v)
                for v in state
            ]

        if torch.is_tensor(state):
            return tuple(state.shape)

        return type(state).__name__

    # =========================================================
    # LOW-LEVEL ACTION
    # =========================================================
    def act(
        self,
        agent,
        obs,
        intent=None,
        hidden=None
    ):

        # -----------------------------------------------------
        # CHECK OBSERVATION
        # -----------------------------------------------------
        obs = np.asarray(
            obs,
            dtype=np.float32
        )

        if obs.shape != (13,):
            raise ValueError(
                f"[LOW ERROR] {agent}: "
                f"Expected observation shape (13,), "
                f"got {obs.shape}"
            )

        # -----------------------------------------------------
        # CURRENT RECURRENT STATE
        # -----------------------------------------------------
        if hidden is None:
            hidden = self.hidden_states.get(
                agent,
                self.module.get_initial_state()
            )

        # -----------------------------------------------------
        # IMPORTANT:
        # Recurrent input = [B, T, OBS]
        #              = [1, 1, 13]
        # -----------------------------------------------------
        obs_tensor = torch.from_numpy(
            obs
        ).reshape(1, 1, 13)

        # -----------------------------------------------------
        # BUILD BATCH
        # -----------------------------------------------------
        batch = {
            "obs": obs_tensor,
            "state_in": self._copy_state(hidden)
        }

        # -----------------------------------------------------
        # RECURRENT SAC INFERENCE
        # -----------------------------------------------------
        with torch.no_grad():

            result = self.module.forward_inference(
                batch
            )

        # -----------------------------------------------------
        # DEBUG OUTPUT
        # -----------------------------------------------------
        if not hasattr(self, "_printed_output_keys"):

            print(
                "\n[LOW DEBUG] RLModule output keys:",
                list(result.keys())
            )

            print(
                "[LOW DEBUG] action_dist_inputs:",
                self._state_shapes(
                    result["action_dist_inputs"]
                )
            )

            print(
                "[LOW DEBUG] state_out:",
                self._state_shapes(
                    result["state_out"]
                )
            )

            self._printed_output_keys = True

        # -----------------------------------------------------
        # UPDATE RECURRENT STATE
        # -----------------------------------------------------
        if "state_out" not in result:
            raise RuntimeError(
                "[LOW ERROR] RLModule did not return "
                "'state_out'."
            )

        next_hidden = self._copy_state(
            result["state_out"]
        )

        self.hidden_states[agent] = self._copy_state(
            next_hidden
        )

        # -----------------------------------------------------
        # ACTION DISTRIBUTION
        # -----------------------------------------------------
        action_dist = (
            self.action_dist_cls.from_logits(
                result["action_dist_inputs"]
            )
        )

        deterministic_dist = (
            action_dist.to_deterministic()
        )

        action = deterministic_dist.sample()

        # -----------------------------------------------------
        # [1,1,3] -> [3]
        # -----------------------------------------------------
        action = (
            action
            .detach()
            .cpu()
            .numpy()
            .reshape(3,)
        )

        action = np.clip(
            action,
            -1.0,
            1.0
        ).astype(np.float32)

        # -----------------------------------------------------
        # DEBUG
        # -----------------------------------------------------
        print(
            f"[LOW] {agent} | "
            f"POS={np.round(obs[0:3], 3)} | "
            f"VEL={np.round(obs[3:6], 3)} | "
            f"TGT={np.round(obs[6:9], 3)} | "
            f"OBS_VEC={np.round(obs[9:12], 3)} | "
            f"OBS_DIST={obs[12]:.3f} | "
            f"ACTION={np.round(action, 4)} | "
            f"STATE={self._state_shapes(next_hidden)}"
        )

        return action, next_hidden
    # =========================================================
    # RESET RECURRENT STATE
    # =========================================================
    def reset_agent(self, agent):

        if agent not in self.agents:
            raise ValueError(
                f"[LOW ERROR] Unknown agent: {agent}"
            )

        initial_state = self.module.get_initial_state()

        self.hidden_states[agent] = self._copy_state(
            initial_state
        )

        print(
            f"[LOW RESET] {agent} | "
            f"state={self._state_shapes(initial_state)}"
        )