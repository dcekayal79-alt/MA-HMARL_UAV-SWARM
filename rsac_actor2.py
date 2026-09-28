import numpy as np
import torch
import torch.nn as nn

from ray.rllib.models.torch.recurrent_net import RecurrentNetwork


class RecurrentSACActor(RecurrentNetwork, nn.Module):

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        nn.Module.__init__(self)

        RecurrentNetwork.__init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
        )

        self.obs_dim = obs_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.hidden_size = 128

        print(
            "[ACTOR INIT] "
            f"obs_dim={self.obs_dim}, "
            f"action_dim={self.action_dim}, "
            f"num_outputs={num_outputs}"
        )

        if self.obs_dim != 13:
            raise ValueError(
                f"Expected 13-D observation, got {self.obs_dim}"
            )

        if self.action_dim != 3:
            raise ValueError(
                f"Expected 3-D action, got {self.action_dim}"
            )

        # ==================================================
        # NETWORK
        # ==================================================

        self.fc1 = nn.Linear(
            self.obs_dim,
            256,
        )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.fc2 = nn.Linear(
            self.hidden_size,
            256,
        )

        # 3 means + 3 log standard deviations
        self.output_layer = nn.Linear(
            256,
            self.action_dim * 2,
        )

        self.activation = nn.ReLU()

    # ======================================================
    # INITIAL STATE
    # ======================================================

    def get_initial_state(self):

        return [
            torch.zeros(
                self.hidden_size,
                dtype=torch.float32,
            ),
            torch.zeros(
                self.hidden_size,
                dtype=torch.float32,
            ),
        ]

    # ======================================================
    # RUNTIME INPUT DIAGNOSTIC
    # ======================================================

    def _diagnose_runtime_input(self, inputs):

        print("\n")
        print("=" * 70)
        print("[RSAC RUNTIME INPUT DIAGNOSTIC]")
        print("=" * 70)

        print(
            "TYPE:",
            type(inputs),
        )

        print(
            "SHAPE:",
            getattr(
                inputs,
                "shape",
                None,
            ),
        )

        print(
            "DTYPE:",
            getattr(
                inputs,
                "dtype",
                None,
            ),
        )

        print(
            "NDIM:",
            getattr(
                inputs,
                "ndim",
                None,
            ),
        )

        print(
            "SIZE:",
            getattr(
                inputs,
                "size",
                None,
            ),
        )

        # --------------------------------------------------
        # NUMPY ARRAY
        # --------------------------------------------------

        if isinstance(
            inputs,
            np.ndarray,
        ):

            print(
                "\nNUMPY ARRAY"
            )

            print(
                "shape =",
                inputs.shape,
            )

            print(
                "dtype =",
                inputs.dtype,
            )

            print(
                "ndim  =",
                inputs.ndim,
            )

            print(
                "size  =",
                inputs.size,
            )

            # ----------------------------------------------
            # OBJECT ARRAY
            # ----------------------------------------------

            if inputs.dtype == np.object_:

                flat = inputs.reshape(-1)

                print(
                    "\nOBJECT ARRAY LENGTH:",
                    len(flat),
                )

                print(
                    "\nOBJECT ARRAY ITEMS:"
                )

                for i, item in enumerate(
                    flat[:10]
                ):

                    print(
                        "\n----------------------------------------"
                    )

                    print(
                        f"ITEM {i}"
                    )

                    print(
                        "type :",
                        type(item),
                    )

                    print(
                        "shape:",
                        getattr(
                            item,
                            "shape",
                            None,
                        ),
                    )

                    print(
                        "dtype:",
                        getattr(
                            item,
                            "dtype",
                            None,
                        ),
                    )

                    print(
                        "ndim :",
                        getattr(
                            item,
                            "ndim",
                            None,
                        ),
                    )

                    print(
                        "size :",
                        getattr(
                            item,
                            "size",
                            None,
                        ),
                    )

                    # --------------------------------------
                    # ITEM IS TORCH TENSOR
                    # --------------------------------------

                    if torch.is_tensor(item):

                        print(
                            "tensor values:",
                            item,
                        )

                        print(
                            "tensor flattened:",
                            item.reshape(-1)[:50],
                        )

                    # --------------------------------------
                    # ITEM IS NUMPY ARRAY
                    # --------------------------------------

                    elif isinstance(
                        item,
                        np.ndarray,
                    ):

                        print(
                            "numpy values:",
                            item,
                        )

                        try:

                            print(
                                "numpy flattened:",
                                item.reshape(-1)[:50],
                            )

                        except Exception:

                            pass

                    # --------------------------------------
                    # OTHER OBJECT
                    # --------------------------------------

                    else:

                        print(
                            "value:",
                            item,
                        )

            else:

                print(
                    "\nNUMPY VALUES:"
                )

                print(
                    inputs.reshape(-1)[:50]
                )

        # --------------------------------------------------
        # TORCH TENSOR
        # --------------------------------------------------

        elif torch.is_tensor(inputs):

            print(
                "\nTORCH TENSOR"
            )

            print(
                "shape =",
                inputs.shape,
            )

            print(
                "dtype =",
                inputs.dtype,
            )

            print(
                "ndim  =",
                inputs.dim(),
            )

            print(
                "numel =",
                inputs.numel(),
            )

            print(
                "\nTORCH VALUES:"
            )

            print(
                inputs.reshape(-1)[:50]
            )

        # --------------------------------------------------
        # LIST / TUPLE
        # --------------------------------------------------

        elif isinstance(
            inputs,
            (list, tuple),
        ):

            print(
                "\nLIST/TUPLE LENGTH:",
                len(inputs),
            )

            for i, item in enumerate(
                inputs[:10]
            ):

                print(
                    "\n----------------------------------------"
                )

                print(
                    f"ITEM {i}"
                )

                print(
                    "type :",
                    type(item),
                )

                print(
                    "shape:",
                    getattr(
                        item,
                        "shape",
                        None,
                    ),
                )

                print(
                    "dtype:",
                    getattr(
                        item,
                        "dtype",
                        None,
                    ),
                )

                print(
                    "value:",
                    item,
                )

        # --------------------------------------------------
        # OTHER
        # --------------------------------------------------

        else:

            print(
                "\nRAW VALUE:"
            )

            print(
                inputs
            )

        print("\n")
        print("=" * 70)
        print(
            "STOPPING INTENTIONALLY FOR DIAGNOSTIC"
        )
        print("=" * 70)

        raise RuntimeError(
            "STOP: inspect actual RSAC runtime input"
        )

    # ======================================================
    # FORWARD
    # ======================================================

    def forward(
        self,
        input_dict,
        state,
        seq_lens,
    ):

        inputs = input_dict["obs"]

        # --------------------------------------------------
        # IMPORTANT
        #
        # We intentionally do NOT convert the input here.
        # We first inspect exactly what RLlib provides.
        # --------------------------------------------------

        self._diagnose_runtime_input(
            inputs
        )

        # This section will not execute.
        # It exists only to keep the model structurally
        # complete.
        #
        # Actual implementation will be restored after
        # the diagnostic output identifies the RLlib input.

        return None, state

    # ======================================================
    # FORWARD RNN
    # ======================================================

        def forward_rnn(
        self,
        inputs,
        state,
        seq_lens,
        ):
            x = self.activation(
            self.fc1(inputs)
            )

            h = state[0]
            c = state[1]

            if h.dim() == 2:
              h = h.unsqueeze(0)

            if c.dim() == 2:
              c = c.unsqueeze(0)

            x, (h_new, c_new) = self.lstm(
              x,
              (h, c)
            )  

            x = self.activation(
              self.fc2(x)
            )

            outputs = self.output_layer(x)

            return outputs, [
              h_new.squeeze(0),
              c_new.squeeze(0),
            ]