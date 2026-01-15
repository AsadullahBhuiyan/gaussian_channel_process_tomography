from contextlib import contextmanager
from datetime import datetime

import joblib
import numpy as np
from thewalrus.symplectic import sympmat
from tqdm.auto import tqdm


class _TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
    """Update tqdm whenever a joblib batch finishes."""

    def __init__(self, tqdm_object, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tqdm_object = tqdm_object

    def __call__(self, *args, **kwargs):
        self.tqdm_object.update(n=self.batch_size)
        if getattr(self.tqdm_object, "_show_datetime", False):
            self.tqdm_object.set_postfix_str(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), refresh=False
            )
        return super().__call__(*args, **kwargs)


@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager linking joblib's callback to a tqdm progress bar."""
    original_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = lambda *args, **kwargs: (
        _TqdmBatchCompletionCallback(tqdm_object, *args, **kwargs)
    )
    try:
        with tqdm_object as pbar:
            yield pbar
    finally:
        joblib.parallel.BatchCompletionCallBack = original_callback


class GaussianChannelLearner:
    """
    A class to simulate, store, and validate the learning of a Gaussian Bosonic Channel.

    The channel is defined by:
        r_out = X @ r_in + d (Drift/Transition)
        Cov_out = X @ Cov_in @ X.T + Y (Noise)

    Units:
        We assume hbar = 2 convention (Vacuum variance = 1).
    """

    def __init__(self, num_modes, true_X=None, true_Y=None):
        """
        Initialize the learner.

        Args:
            num_modes (int): Number of modes in the system (N).
            true_X (np.array): (Optional) The ground truth X matrix (2N x 2N) for simulation.
            true_Y (np.array): (Optional) The ground truth Y matrix (2N x 2N) for simulation.
        """
        self.N = num_modes
        self.dim = 2 * num_modes
        self.hbar = 2.0  # Convention: Vacuum noise variance = 1

        # Storage for the estimated channel matrices
        self.est_X = None
        self.est_Y = None

        # Ground truth for simulation (if provided)
        self.true_X = true_X
        self.true_Y = true_Y

        # Data storage
        self.probe_inputs = []
        self.measured_means = []
        self.measured_covs = []

    def _simulate_channel_action(self, mean_in, cov_in):
        """
        Internal method to simulate the channel action on a state using Ground Truth.
        """
        if self.true_X is None or self.true_Y is None:
            raise ValueError("Ground truth X/Y matrices not set. Cannot simulate data.")

        # 1. Apply Drift: r_out = X * r_in
        mean_out = self.true_X @ mean_in

        # 2. Apply Noise: Sig_out = X * Sig_in * X^T + Y
        cov_out = self.true_X @ cov_in @ self.true_X.T + self.true_Y

        return mean_out, cov_out

    def run_drift_protocol(self, alpha=1.0):
        """
        STEP 1: Learn X.
        Protocol: Displace the vacuum to create unit coherent states along
        every basis vector in phase space.
        """
        print("--- Starting Drift Protocol (Learning X) ---")

        # Create input matrix R_in (columns are input vectors)
        # We will use orthogonal probes scaled by alpha
        R_in = np.eye(self.dim) * alpha
        R_out_cols = []

        for k in tqdm(range(self.dim), desc="Drift probes"):
            # Input vector: k-th column of R_in
            r_in = R_in[:, k]

            # Input Covariance: Coherent state (Vacuum noise)
            cov_in = np.eye(self.dim) * (self.hbar / 2)

            # Simulate measurement
            # In a real experiment, this 'mean_out' comes from averaging homodyne samples
            mean_out, _ = self._simulate_channel_action(r_in, cov_in)

            R_out_cols.append(mean_out)

        # Stack results into a matrix
        R_out = np.column_stack(R_out_cols)

        # SOLVE: X = R_out @ R_in^-1
        # Since R_in is diagonal (alpha * I), inverse is just (1/alpha) * I
        self.est_X = R_out @ np.linalg.inv(R_in)

        print(f"Drift matrix X estimated using {self.dim} probes.")
        return self.est_X

    def run_noise_protocol(self):
        """
        STEP 2: Learn Y.
        Protocol: Send in a Vacuum state (zero mean) and measure output covariance.
        """
        print("--- Starting Noise Protocol (Learning Y) ---")

        if self.est_X is None:
            raise RuntimeError("Must run drift_protocol (Step 1) before noise_protocol.")

        # Input: Vacuum State
        r_in = np.zeros(self.dim)
        cov_in = np.eye(self.dim) * (self.hbar / 2)  # Shot noise units

        # Simulate measurement
        # In experiment: You perform homodyne tomography here to reconstruct Sig_out
        _, cov_out = self._simulate_channel_action(r_in, cov_in)

        # SOLVE: Y = Sig_out - X @ Sig_in @ X^T
        # We use our *estimated* X from Step 1
        noise_from_amplification = self.est_X @ cov_in @ self.est_X.T
        self.est_Y = cov_out - noise_from_amplification

        print("Noise matrix Y estimated.")
        return self.est_Y

    def validate_cp_condition(self):
        """
        STEP 3: Validation.
        Check if the estimated channel is physically valid (Complete Positivity).

        Condition: Y + i*Omega - i*X*Omega*X^T >= 0
        """
        print("--- Validating Physicality (CP Condition) ---")

        if self.est_X is None or self.est_Y is None:
            raise RuntimeError("Model not fully trained.")

        # 1. Get Symplectic Matrix Omega using thewalrus
        Omega = sympmat(self.N)

        # 2. Construct the test matrix M
        # M = Y + i*Omega - i * X @ Omega @ X.T
        term1 = self.est_Y
        term2 = 1j * Omega
        term3 = 1j * (self.est_X @ Omega @ self.est_X.T)

        M = term1 + term2 - term3

        # 3. Check Eigenvalues (must be >= 0)
        # Since M is Hermitian, eigenvalues are real.
        eigvals = np.linalg.eigvalsh(M)
        min_eig = np.min(eigvals)

        is_valid = min_eig >= -1e-9  # allow small float tolerance

        print(f"Minimum Eigenvalue of CP matrix: {min_eig:.4e}")
        if is_valid:
            print("SUCCESS: Channel is physically valid.")
        else:
            print("WARNING: Channel violates uncertainty principle (unphysical).")
            print("This usually means Y is too small for the amount of squeezing in X.")

        return is_valid
