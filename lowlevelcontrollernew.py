import numpy as np
import torch

from ray.rllib.algorithms.sac import SAC


class LowLevelController:

    def __init__(self, agents):

        self.agents = agents

        # ============================================================
        # LOAD TRAINED RSAC CHECKPOINT
        # ============================================================
        print("\nLoading RSAC checkpoint...")

        self.algo = SAC.from_checkpoint(
            "E:/uav_hmarl_project/sac_checkpoint_final"
        )

        # ============================================================
        # GET RL MODULE
        # ============================================================
        self.module = self.algo.get_module()

        print("\n========== RL MODULE DEBUG ==========")
        print("Module:", type(self.module))

        try:
            print(
                "Is stateful:",
                self.module.is_stateful()
            )
        except Exception as e:
            print(
                "Is stateful check failed:",
                e
            )

        try:
            initial_state = self.module.get_initial_state()
            print(
                "Initial state:",
                initial_state
            )
        except Exception as e:
            print(
                "Initial state check failed:",
                e
            )

        try:
            self.action_dist_cls = (
                self.module.get_inference_action_dist_cls()
            )

            print(
                "Action distribution class:",
                self.action_dist_cls
            )

        except Exception as e:

            print(
                "Action distribution class ERROR:",
                e
            )

            raise

        print("====================================\n")

        # ============================================================
        # PREVIOUS ACTION / REWARD
        # ============================================================
        self.prev_actions = {
            agent: np.zeros(
                3,
                dtype=np.float32
            )
            for agent in agents
        }

        self.prev_rewards = {
            agent: 0.0
            for agent in agents
        }

    # ================================================================
    # RESET
    # ================================================================
    def reset(self, agent=None):

        if agent is not None:

            self.prev_actions[agent] = np.zeros(
                3,
                dtype=np.float32
            )

            self.prev_rewards[agent] = 0.0

        else:

            for agent_name in self.agents:

                self.prev_actions[agent_name] = np.zeros(
                    3,
                    dtype=np.float32
                )

                self.prev_rewards[agent_name] = 0.0

    # ================================================================
    # LOW-LEVEL ACTION
    # ================================================================
    def act(
        self,
        agent,
        obs,
        intent=None,
        hidden=None
    ):

        # ------------------------------------------------------------
        # Convert observation
        # ------------------------------------------------------------
        obs = np.asarray(
            obs,
            dtype=np.float32
        )

        # ------------------------------------------------------------
        # IMPORTANT:
        #
        # The trained RSAC environment has a 13-D observation.
        #
        # Do NOT concatenate:
        #
        # obs + intent
        #
        # because that would change the input from 13-D.
        # ------------------------------------------------------------
        if obs.shape != (13,):

            raise ValueError(
                f"RSAC expected 13-D observation, "
                f"but received shape {obs.shape}"
            )

        # ------------------------------------------------------------
        # Convert observation to batch tensor
        # ------------------------------------------------------------
        obs_tensor = torch.from_numpy(
            obs
        ).unsqueeze(0)

        # ------------------------------------------------------------
        # RLModule inference
        # ------------------------------------------------------------
        with torch.no_grad():

            result = self.module.forward_inference(
                {
                    "obs": obs_tensor
                }
            )

        # ------------------------------------------------------------
        # Get action distribution inputs
        # ------------------------------------------------------------
        action_dist_inputs = result[
            "action_dist_inputs"
        ]

        # ------------------------------------------------------------
        # Construct SAC action distribution
        # ------------------------------------------------------------
        action_dist = self.action_dist_cls.from_logits(
            action_dist_inputs
        )

        # ------------------------------------------------------------
        # Inference must be deterministic.
        #
        # This corresponds to using the mean/greedy action rather
        # than randomly sampling from SAC's policy.
        # ------------------------------------------------------------
        deterministic_dist = action_dist.to_deterministic()

        action = deterministic_dist.sample()

        # ------------------------------------------------------------
        # Convert to NumPy
        # ------------------------------------------------------------
        action = action[0].detach().cpu().numpy()

        action = np.asarray(
            action,
            dtype=np.float32
        )

        # ------------------------------------------------------------
        # Ensure correct 3-D action
        # ------------------------------------------------------------
        if action.shape != (3,):

            raise ValueError(
                f"RSAC produced unexpected action shape "
                f"{action.shape}; expected (3,)"
            )

        # ------------------------------------------------------------
        # Safety clipping
        # ------------------------------------------------------------
        action = np.clip(
            action,
            -1.0,
            1.0
        ).astype(np.float32)

        # ------------------------------------------------------------
        # Store previous action
        # ------------------------------------------------------------
        self.prev_actions[agent] = action.copy()

        # ------------------------------------------------------------
        # CURRENT RLModule IS NOT STATEFUL
        #
        # Therefore there is no LSTM hidden state to return from
        # this new RLModule.
        #
        # fullhmarlnew.py expects:
        #
        # action, next_hidden = low.act(...)
        #
        # so return the incoming hidden state unchanged.
        # ------------------------------------------------------------
        next_hidden = hidden
        print(
            f"[LOW] {agent} | "
            f"POS={np.round(obs[0:3], 3)} | "
            f"VEL={np.round(obs[3:6], 3)} | "
            f"TGT={np.round(obs[6:9], 3)} | "
            f"OBS_VEC={np.round(obs[9:12], 3)} | "
            f"OBS_DIST={obs[12]:.3f} | "
            f"ACTION={np.round(action, 4)}"
        )

        return action, next_hidden