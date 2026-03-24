# helpers/nn_model.py
"""Physics-informed neural network model for bending-wave amplitude estimation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_phase(r, k_rad):
    """Integrates wavenumber k(r) to produce cumulative phase using the trapezoidal rule.

    .. code-block:: text

        psi[0] = 0
        psi[i] = psi[i-1] + 0.5 * (k[i] + k[i-1]) * (r[i] - r[i-1])

    Args:
        r (torch.Tensor): 1-D tensor of radial positions in km.
        k_rad (torch.Tensor): 1-D tensor of wavenumbers in rad/km, same
            length as ``r``.

    Returns:
        torch.Tensor: 1-D tensor of cumulative phase in radians, same length
            as ``r``.  The first element is always 0.
    """
    dr = torch.diff(r)
    incr = 0.5 * (k_rad[1:] + k_rad[:-1]) * dr
    psi = torch.zeros_like(r)
    psi[1:] = torch.cumsum(incr, dim=0)
    return psi


def build_features(r_t, k_t, cot_t):
    """Builds per-radius input features for the bending-wave neural network.

    All three inputs must be 1-D tensors of the same length ``N``.  Each
    quantity is z-score normalized before being stacked.  The five features
    are:

    1. Normalized radius
    2. Normalized wavenumber
    3. Normalized cotangent of elevation angle
    4. Radial gradient of wavenumber ``dk/dr``
    5. Inverse wavenumber ``1 / |k|``

    Args:
        r_t (torch.Tensor): 1-D tensor of radial positions in km.
        k_t (torch.Tensor): 1-D tensor of wavenumbers in rad/km.
        cot_t (torch.Tensor): 1-D tensor of cotangent of effective elevation
            angle ``cot(B_eff)``.

    Returns:
        tuple:
            - **X** (torch.Tensor): Feature matrix of shape ``[N, 5]``.
            - **psi** (torch.Tensor): Cumulative phase in radians from
              :func:`build_phase`, shape ``[N]``.
    """
    r_n = (r_t - r_t.mean()) / (r_t.std() + 1e-6)
    k_n = (k_t - k_t.mean()) / (k_t.std() + 1e-6)
    cot_n = (cot_t - cot_t.mean()) / (cot_t.std() + 1e-6)

    psi = build_phase(r_t, k_t)

    dk = torch.zeros_like(k_t)
    dk[1:] = (k_t[1:] - k_t[:-1]) / (r_t[1:] - r_t[:-1] + 1e-6)

    inv_k = 1.0 / (k_t.abs() + 1e-3)

    X = torch.stack(
        [r_n, k_n, cot_n, dk, inv_k],
        dim=1
    )
    return X, psi


class ContextAmpCSNet(nn.Module):
    """1-D convolutional network that predicts complex bending-wave amplitude.

    Outputs per-radius cosine and sine amplitude components ``A_c(r)`` and
    ``A_s(r)`` such that the physical vertical amplitude is

    .. code-block:: text

        A_V(r) = sqrt(A_c(r)^2 + A_s(r)^2)

    and the photometric model is

    .. code-block:: text

        IF(r) = IF0 * [1 - cotB(r) * k(r) * (A_c(r)*cos(psi) + A_s(r)*sin(psi))]

    Architecture: two 1-D conv layers (kernel=5, same padding) followed by
    two fully-connected layers producing the two output channels.

    Attributes:
        conv1 (nn.Conv1d): First convolutional layer (``in_dim`` → ``hidden``).
        conv2 (nn.Conv1d): Second convolutional layer (``hidden`` → ``hidden``).
        fc1 (nn.Linear): First fully-connected layer (``hidden`` → ``hidden``).
        fc2 (nn.Linear): Output layer (``hidden`` → 2).
    """

    def __init__(self, in_dim=5, hidden=64):
        """Initializes ContextAmpCSNet.

        Args:
            in_dim (int): Number of input features per radius point.
                Defaults to ``5``.
            hidden (int): Number of channels in the convolutional and
                hidden fully-connected layers. Defaults to ``64``.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, 2)   # → A_c, A_s

    def forward(self, x):
        """Forward pass of the network.

        Args:
            x (torch.Tensor): Feature tensor of shape ``[N, F]`` where ``N``
                is the number of radius points and ``F`` is the feature
                dimension (``in_dim``).

        Returns:
            tuple:
                - **A_c** (torch.Tensor): Cosine amplitude component,
                  shape ``[N]``.
                - **A_s** (torch.Tensor): Sine amplitude component,
                  shape ``[N]``.
        """
        # x: [N, F]
        x = x.unsqueeze(0).transpose(1, 2)    # → [1, F, N]
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))             # [1, H, N]
        h = h.transpose(1, 2).squeeze(0)      # → [N, H]
        h = F.relu(self.fc1(h))
        out = self.fc2(h)                     # → [N, 2]
        A_c = out[:, 0]
        A_s = out[:, 1]
        return A_c, A_s


def forward_physics(r, k_rad, cotB, A_c, A_s, IF0=1.0):
    """Computes the photometric I/F model from predicted amplitude components.

    Implements the bending-wave photometric equation:

    .. code-block:: text

        IF(r) = IF0 * [1 - cotB(r) * k(r) * (A_c(r)*cos(psi(r)) + A_s(r)*sin(psi(r)))]

    where ``psi(r)`` is the cumulative phase from :func:`build_phase`.
    The physical vertical amplitude is related by

    .. code-block:: text

        A_V(r) = sqrt(A_c(r)^2 + A_s(r)^2)

    Args:
        r (torch.Tensor): 1-D tensor of radial positions in km.
        k_rad (torch.Tensor): 1-D tensor of wavenumbers in rad/km.
        cotB (torch.Tensor): 1-D tensor of cotangent of effective elevation
            angle.
        A_c (torch.Tensor): 1-D tensor of cosine amplitude components
            predicted by the network.
        A_s (torch.Tensor): 1-D tensor of sine amplitude components
            predicted by the network.
        IF0 (float): Baseline I/F reference level. Defaults to ``1.0``.

    Returns:
        torch.Tensor: 1-D tensor of modeled I/F values, same length as ``r``.
    """
    psi = build_phase(r, k_rad)
    cos_psi = torch.cos(psi)
    sin_psi = torch.sin(psi)

    oscillation = A_c * cos_psi + A_s * sin_psi

    IF_model = IF0 * (1.0 - cotB * k_rad * oscillation)
    return IF_model
