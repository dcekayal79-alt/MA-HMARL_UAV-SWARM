import torch
import torch.nn as nn

from ray.rllib.models.torch.recurrent_net import RecurrentNetwork


class RecurrentSACQModel(RecurrentNetwork, nn.Module):
    """
    Recurrent Q-network for SAC.

    Input:
        13-D observation + 3-D action = 16-D

    Architecture:
        16
        ↓
        FC 256
        ↓
        LSTM 128
        ↓
        FC 256
        ↓
        Q value

    Output:
        Single scalar Q-value
    """

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
            1,
            model_config,
            name,
        )

        self.input_dim = obs_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.hidden_size = 128

        if self.input_dim != 16:
            raise ValueError(
                f"Expected 16-D Q input (13 obs + 3 action), "
                f"got {self.input_dim}"
            )

        if self.action_dim != 3:
            raise ValueError(
                f"Expected 3-D action, got {self.action_dim}"
            )

        self.fc1 = nn.Linear(
            self.input_dim,
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

        self.q_output = nn.Linear(
            256,
            1,
        )

        self.activation = nn.ReLU()

    def get_initial_state(self):

        return [
            torch.zeros(self.hidden_size),
            torch.zeros(self.hidden_size),
        ]

    def forward(
        self,
        input_dict,
        state,
        seq_lens,
    ):
        """
        Forward pass for RLlib SAC.

        Handles:
        - seq_lens=None
        - single observations
        - batched observations
        - RLlib dummy loss calls
        - recurrent hidden-state batch expansion
        """

        inputs = input_dict["obs"]

        # --------------------------------------------------
        # Convert input to tensor
        # --------------------------------------------------

        if not torch.is_tensor(inputs):
            inputs = torch.as_tensor(
                inputs,
                dtype=torch.float32,
            )

        inputs = inputs.float()

        # --------------------------------------------------
        # Ensure batch dimension
        # --------------------------------------------------

        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)

        # --------------------------------------------------
        # Validate input dimension
        # --------------------------------------------------

        if inputs.shape[-1] != 16:
            raise ValueError(
                f"Q-network received {inputs.shape[-1]} inputs; "
                f"expected 16 (13 obs + 3 action)"
            )

        batch_size = inputs.shape[0]

        # --------------------------------------------------
        # Get recurrent state
        # --------------------------------------------------

        if not state:
            state = self.get_initial_state()

        h = state[0]
        c = state[1]

        # --------------------------------------------------
        # Convert hidden state to tensor
        # --------------------------------------------------

        if not torch.is_tensor(h):
            h = torch.as_tensor(
                h,
                dtype=torch.float32,
                device=inputs.device,
            )

        if not torch.is_tensor(c):
            c = torch.as_tensor(
                c,
                dtype=torch.float32,
                device=inputs.device,
            )

        h = h.to(
            device=inputs.device,
            dtype=torch.float32,
        )

        c = c.to(
            device=inputs.device,
            dtype=torch.float32,
        )

        # --------------------------------------------------
        # Ensure hidden state has batch dimension
        # --------------------------------------------------

        if h.dim() == 1:
            h = h.unsqueeze(0)

        if c.dim() == 1:
            c = c.unsqueeze(0)

        # --------------------------------------------------
        # Expand hidden state to actual batch size
        #
        # RLlib may provide:
        #     h = [1, 128]
        #
        # while the input batch is:
        #     batch_size = 32
        #
        # LSTM requires:
        #     [1, 32, 128]
        # --------------------------------------------------

        if h.shape[0] == 1 and batch_size > 1:
            h = h.expand(
                batch_size,
                -1,
            ).contiguous()

        if c.shape[0] == 1 and batch_size > 1:
            c = c.expand(
                batch_size,
                -1,
            ).contiguous()

        # --------------------------------------------------
        # FC layer
        # --------------------------------------------------

        x = self.activation(
            self.fc1(inputs)
        )

        # --------------------------------------------------
        # Add sequence dimension
        #
        # Current shape:
        #     [batch, 256]
        #
        # Required by LSTM:
        #     [batch, sequence, 256]
        # --------------------------------------------------

        x = x.unsqueeze(1)

        # --------------------------------------------------
        # Add LSTM layer dimension
        #
        # Current:
        #     [batch, 128]
        #
        # Required:
        #     [1, batch, 128]
        # --------------------------------------------------

        if h.dim() == 2:
            h = h.unsqueeze(0)

        if c.dim() == 2:
            c = c.unsqueeze(0)

        # --------------------------------------------------
        # LSTM
        # --------------------------------------------------

        x, (h_new, c_new) = self.lstm(
            x,
            (h, c),
        )

        # --------------------------------------------------
        # FC layer
        # --------------------------------------------------

        x = self.activation(
            self.fc2(x)
        )

        # --------------------------------------------------
        # Q-value
        # --------------------------------------------------

        q_value = self.q_output(x)

        # [batch, 1, 1] → [batch, 1]
        q_value = q_value.squeeze(1)

        # --------------------------------------------------
        # Return
        # --------------------------------------------------

        return q_value, [
            h_new.squeeze(0),
            c_new.squeeze(0),
        ]

    def forward_rnn(
        self,
        inputs,
        state,
        seq_lens,
    ):
        """
        RecurrentNetwork interface.

        Used when RLlib provides sequence batches directly.
        """

        # --------------------------------------------------
        # FC layer
        # --------------------------------------------------

        x = self.activation(
            self.fc1(inputs)
        )

        # --------------------------------------------------
        # Hidden state
        # --------------------------------------------------

        h = state[0]
        c = state[1]

        if h.dim() == 2:
            h = h.unsqueeze(0)

        if c.dim() == 2:
            c = c.unsqueeze(0)

        # --------------------------------------------------
        # LSTM
        # --------------------------------------------------

        x, (h_new, c_new) = self.lstm(
            x,
            (h, c),
        )

        # --------------------------------------------------
        # FC layer
        # --------------------------------------------------

        x = self.activation(
            self.fc2(x)
        )

        # --------------------------------------------------
        # Q-value
        # --------------------------------------------------

        q_value = self.q_output(x)

        return q_value, [
            h_new.squeeze(0),
            c_new.squeeze(0),
        ]