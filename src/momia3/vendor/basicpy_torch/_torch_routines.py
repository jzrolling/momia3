"""
PyTorch implementation of fitting routines converted from JAX.

This file contains the PyTorch implementation of the fitting routines that were
originally written in JAX. The conversion was assisted by doubao-seed-2.0-Code.

Key changes from JAX to PyTorch:
- Replaced JAX jit with PyTorch eager execution
- Replaced JAX lax.while_loop with Python loops
- Replaced JAX array operations with PyTorch tensor operations
- Added device handling for CPU/CUDA/MPS compatibility
- Maintained the same algorithmic logic and mathematical formulas
"""

from functools import partial
from typing import Tuple

import numpy as np
import torch
from pydantic import BaseModel, Field

from .tools.dct_tools import TorchDCT

idct2d, dct2d, idct3d, dct3d = TorchDCT.idct2d, TorchDCT.dct2d, TorchDCT.idct3d, TorchDCT.dct3d


def _jshrinkage(x, thresh):
    return torch.sign(x) * torch.maximum(torch.abs(x) - thresh, torch.tensor(0.0, device=x.device))


class BaseFit(BaseModel):
    epsilon: float = Field(
        0.1,
        description="Weight regularization term.",
    )
    max_mu: float = Field(0, description="The maximum value of mu.")
    init_mu: float = Field(0, description="Initial value for mu.")
    D_Z_max: float = Field(0, description="Maximum value for D_Z.")
    image_norm: float = Field(0, description="The 2nd order norm for the images.")
    rho: float = Field(1.5, description="Parameter rho for mu update.")
    optimization_tol: float = Field(
        1e-6,
        description="Optimization tolerance.",
    )
    optimization_tol_diff: float = Field(
        1e-6,
        description="Optimization tolerance for update diff.",
    )
    smoothness_darkfield: float = Field(
        0.0,
        description="Darkfield smoothness weight for sparse reguralization.",
    )
    sparse_cost_darkfield: float = Field(
        0.0,
        description="Darkfield sparseness weight for sparse reguralization.",
    )
    smoothness_flatfield: float = Field(
        0.0,
        description="Flatfield smoothness weight for sparse reguralization.",
    )
    get_darkfield: bool = Field(
        False,
        description="When True, will estimate the darkfield shading component.",
    )
    max_iterations: int = Field(
        500,
        description="Maximum number of iterations for single optimization.",
    )

    class Config:
        frozen = True
        extra = "ignore"

    def _cond(self, vals):
        k = vals[0]
        fit_residual = vals[-2]
        value_diff = vals[-1]
        norm_ratio = torch.linalg.norm(fit_residual.flatten(), ord=2) / self.image_norm
        conv = torch.any(
            torch.tensor(
                [
                    norm_ratio > self.optimization_tol,
                    value_diff > self.optimization_tol_diff,
                ]
            )
        )
        return torch.all(
            torch.tensor(
                [
                    conv,
                    k < self.max_iterations,
                ]
            )
        )

    def _fit_jit(
        self,
        Im,
        W,
        W_D,
        S,
        D_R,
        D_Z,
        B,
        I_R,
        progress_callback=None,
    ):
        # initialize values
        Y = torch.zeros_like(Im, dtype=torch.float32, device=Im.device)
        mu = torch.tensor(self.init_mu, dtype=torch.float32, device=Im.device)
        fit_residual = torch.ones(Im.shape, dtype=torch.float32, device=Im.device) * torch.inf
        value_diff = torch.inf

        vals = (0, S, D_R, D_Z, I_R, B, Y, mu, fit_residual, value_diff)
        step = partial(
            self._step,
            Im,
            W,
            W_D,
        )
        
        # PyTorch doesn't have a direct equivalent of lax.while_loop, so we use a for loop
        # with a break condition
        for i in range(self.max_iterations):
            if not self._cond(vals):
                break
            vals = step(vals)
            """
            # Report progress if callback is provided
            if progress_callback is not None:
                # This is an inner loop, so we don't update progress for every iteration
                # Instead, we update every 10 iterations to avoid overhead
                if (i + 1) % 10 == 0 or (i + 1) == self.max_iterations:
                    progress_callback(i / self.max_iterations)
            """
        k, S, D_R, D_Z, I_R, B, Y, mu, fit_residual, value_diff = vals
        norm_ratio = torch.linalg.norm(fit_residual.flatten(), ord=2) / self.image_norm
        return S, D_R, D_Z, I_R, B, norm_ratio.item(), k < self.max_iterations

    def _fit_baseline_jit(
        self,
        Im,
        W,
        S,
        D,
        B,
        I_R,
        progress_callback=None,
    ):
        # initialize values
        Y = torch.zeros_like(Im, dtype=torch.float32, device=Im.device)
        mu = torch.tensor(self.init_mu, dtype=torch.float32, device=Im.device)
        fit_residual = torch.ones(Im.shape, dtype=torch.float32, device=Im.device) * torch.inf
        value_diff = torch.inf

        vals = (0, I_R, B, Y, mu, fit_residual, value_diff)
        step = partial(
            self._step_only_baseline,
            Im,
            W,
            S,
            D,
        )

        # PyTorch doesn't have a direct equivalent of lax.while_loop, so we use a for loop
        # with a break condition
        for i in range(self.max_iterations):
            if not self._cond(vals):
                break
            vals = step(vals)
            
            """
            # Report progress if callback is provided
            if progress_callback is not None:
                # This is an inner loop, so we don't update progress for every iteration
                # Instead, we update every 10 iterations to avoid overhead
                if (i + 1) % 10 == 0 or (i + 1) == self.max_iterations:
                    progress_callback(i / self.max_iterations)
            """
        k, I_R, B, Y, mu, fit_residual, value_diff = vals
        norm_ratio = torch.linalg.norm(fit_residual.flatten(), ord=2) / self.image_norm
        return I_R, B, norm_ratio.item(), k < self.max_iterations

    def fit(
        self,
        Im,
        W,
        W_D,
        S,
        D_R,
        D_Z,
        B,
        I_R,
        progress_callback=None,
    ):
        if S.shape != Im.shape[1:]:
            raise ValueError("S must have the same shape as images.shape[1:]")
        if D_R.shape != Im.shape[1:]:
            raise ValueError("D_R must have the same shape as images.shape[1:]")
        if isinstance(D_Z, torch.Tensor):
            if D_Z.numel() != 1:
                raise ValueError("D_Z must be a scalar.")
        elif not np.isscalar(D_Z):
            raise ValueError("D_Z must be a scalar.")
        if B.shape != Im.shape[:1]:
            raise ValueError("B must have the same shape as images.shape[:1]")
        if I_R.shape != Im.shape:
            raise ValueError("I_R must have the same shape as images.shape")
        if W.shape != Im.shape:
            raise ValueError("weight must have the same shape as images.shape")
        if W_D.shape != Im.shape[1:]:
            raise ValueError(
                "darkfield weight must have the same shape as images.shape[1:"
            )
        return self._fit_jit(Im, W, W_D, S, D_R, D_Z, B, I_R, progress_callback=progress_callback)

    def fit_baseline(
        self,
        Im,
        W,
        S,
        D,
        B,
        I_R,
        progress_callback=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, bool]:
        if S.shape != Im.shape[1:]:
            raise ValueError("S must have the same shape as images.shape[1:]")
        if D.shape != Im.shape[1:]:
            raise ValueError("D must have the same shape as images.shape[1:]")
        if B.shape != Im.shape[:1]:
            raise ValueError("B must have the same shape as images.shape[:1]")
        if I_R.shape != Im.shape:
            raise ValueError("I_R must have the same shape as images.shape")
        if W.shape != Im.shape:
            raise ValueError("weight must have the same shape as images.shape")
        return self._fit_baseline_jit(Im, W, S, D, B, I_R, progress_callback=progress_callback)


class LadmapFit(BaseFit):
    def _step(
        self,
        Im,
        weight,
        dark_weight,
        vals,
    ):
        k, S, D_R, D_Z, I_R, B, Y, mu, fit_residual, value_diff = vals
        T_max = Im.shape[0]

        I_B = S.unsqueeze(0) * B.view(-1, 1, 1, 1) + D_R.unsqueeze(0) + D_Z
        eta_S = torch.sum(B**2) * 1.02 + 0.01
        S_new = (
            S
            + torch.sum(B.view(-1, 1, 1, 1) * (Im - I_B - I_R + Y / mu), dim=0)
            / eta_S
        )
        S_new = idct3d(
            _jshrinkage(dct3d(S_new), self.smoothness_flatfield / (eta_S * mu))
        )
        S_new = torch.where(S_new.min() < 0, S_new - S_new.min(), S_new)
        dS = S_new - S
        S = S_new

        I_B = S.unsqueeze(0) * B.view(-1, 1, 1, 1) + D_R.unsqueeze(0) + D_Z
        I_R_new = _jshrinkage(Im - I_B + Y / mu, weight / mu / T_max)
        dI_R = I_R_new - I_R
        I_R = I_R_new

        R = Im - I_R
        S_sq = torch.sum(S**2)
        B_new = torch.sum(S.unsqueeze(0) * (R + Y / mu), dim=(1, 2, 3)) / S_sq
        B_new = torch.where(S_sq > 0, B_new, B)
        B_new = torch.maximum(B_new, torch.tensor(0.0, device=B_new.device))

        mean_B = torch.mean(B_new)
        B_new = torch.where(mean_B > 0, B_new / mean_B, B_new)
        S = torch.where(mean_B > 0, S * mean_B, S)

        dB = B_new - B
        B = B_new

        BS = S.unsqueeze(0) * B.view(-1, 1, 1, 1)
        if self.get_darkfield:
            D_Z_new = torch.mean(Im - BS - D_R.unsqueeze(0) - I_R + Y / 2.0 / mu)
            D_Z_new = torch.clamp(D_Z_new, 0, self.D_Z_max)
            dD_Z = D_Z_new - D_Z
            D_Z = D_Z_new

            eta_D = Im.shape[0] * 1.02
            D_R_new = D_R + 1.0 / eta_D * torch.sum(
                Im - BS - D_R.unsqueeze(0) - D_Z - I_R + Y / mu, dim=0
            )
            D_R_new = idct3d(
                _jshrinkage(dct3d(D_R_new), self.smoothness_darkfield / eta_D / mu)
            )
            D_R_new = _jshrinkage(
                D_R_new, self.sparse_cost_darkfield * dark_weight / eta_D / mu
            )
            dD_R = D_R_new - D_R
            D_R = D_R_new

        I_B = BS + D_R.unsqueeze(0) + D_Z
        fit_residual = R - I_B
        Y = Y + mu * fit_residual

        value_diff = torch.max(
            torch.tensor(
                [
                    torch.linalg.norm(dS.flatten(), ord=2) * torch.sqrt(eta_S),
                    torch.linalg.norm(dI_R.flatten(), ord=2) * torch.sqrt(torch.tensor(1.0)),
                    # TODO find better form with theoretical evidence
                    torch.linalg.norm(dB.flatten(), ord=2),
                ]
            )
        )

        if self.get_darkfield:
            value_diff = torch.max(
                torch.tensor(
                    [
                        value_diff,
                        torch.linalg.norm(dD_R.flatten(), ord=2) * torch.sqrt(torch.as_tensor(float(eta_D), dtype=torch.float32, device=dD_R.device)),
                        # TODO find better form with theoretical evidence
                        dD_Z**2,
                    ]
                )
            )
        value_diff = value_diff / self.image_norm
        mu = torch.minimum(mu * self.rho, torch.tensor(self.max_mu, dtype=torch.float32, device=S.device))

        return (k + 1, S, D_R, D_Z, I_R, B, Y, mu, fit_residual, value_diff)

    def _step_only_baseline(self, Im, weight, S, D, vals):
        k, I_R, B, Y, mu, fit_residual, value_diff = vals
        T_max = Im.shape[0]

        I_B = S.unsqueeze(0) * B.view(-1, 1, 1, 1) + D.unsqueeze(0)
        I_R_new = _jshrinkage(Im - I_B + Y / mu, weight / mu / T_max)
        dI_R = I_R_new - I_R
        I_R = I_R_new

        R = Im - I_R
        B_new = torch.sum(S.unsqueeze(0) * (R + Y / mu), dim=(1, 2, 3)) / torch.sum(S**2)
        B_new = torch.maximum(B_new, torch.tensor(0.0, device=B_new.device))
        dB = B_new - B
        B = B_new

        I_B = S.unsqueeze(0) * B.view(-1, 1, 1, 1) + D.unsqueeze(0)
        fit_residual = R - I_B
        Y = Y + mu * fit_residual

        value_diff = torch.max(
            torch.tensor(
                [
                    torch.linalg.norm(dI_R.flatten(), ord=2) * torch.sqrt(torch.tensor(1.0)),
                    # TODO find better form with theoretical evidence
                    torch.linalg.norm(dB.flatten(), ord=2),
                ]
            )
        )
        value_diff = value_diff / self.image_norm

        mu = torch.minimum(mu * self.rho, torch.tensor(self.max_mu, dtype=torch.float32, device=I_R.device))

        return (k + 1, I_R, B, Y, mu, fit_residual, value_diff)

    def calc_weights(self, I_B, I_R):
        Ws = torch.ones_like(I_R, dtype=torch.float32) / (
            torch.abs(I_R / (I_B + self.epsilon)) + self.epsilon
        )
        return Ws / torch.mean(Ws)

    def calc_dark_weights(self, D_R):
        Ws = torch.ones_like(D_R, dtype=torch.float32) / (torch.abs(D_R) + self.epsilon)
        return Ws / torch.mean(Ws)

    def calc_weights_baseline(self, I_B, I_R):
        return self.calc_weights(I_B, I_R)

    def calc_darkfield(_self, S, D_R, D_Z):
        return D_R + D_Z


class ApproximateFit(BaseFit):
    def _step(
        self,
        Im,
        weight,
        dark_weight,
        vals,
    ):
        k, S, D_R, D_Z, I_R, B, Y, mu, fit_residual, value_diff = vals

        # approximate fitting only accepts two-dimensional images.
        # XXX better coding?
        Im = Im[:, 0, ...]
        weight = weight[:, 0, ...]
        S = S[0]
        D_R = D_R[0]
        I_R = I_R[:, 0, ...]
        Y = Y[:, 0, ...]

        S_hat = dct2d(S)
        I_B = S.unsqueeze(0) * B.view(-1, 1, 1) + D_R.unsqueeze(0) + D_Z
        temp_W = (Im - I_B - I_R + Y / mu) / 1.0
        temp_W = torch.mean(temp_W, dim=0)
        S_hat = S_hat + dct2d(temp_W)
        S_hat = _jshrinkage(S_hat, self.smoothness_flatfield / (1.0 * mu))
        S = idct2d(S_hat)
        I_B = S.unsqueeze(0) * B.view(-1, 1, 1) + D_R.unsqueeze(0) + D_Z
        I_R = (Im - I_B + Y / mu) / 1.0
        I_R = _jshrinkage(I_R, weight / (1.0 * mu))
        R = Im - I_R
        B = torch.mean(R, dim=(1, 2)) / torch.mean(R)
        B = torch.maximum(B, torch.tensor(0.0, device=B.device))

        if self.get_darkfield:
            B_valid = B < 1

            S_inmask = S > torch.mean(S) * (1 - 1e-6)
            S_outmask = S < torch.mean(S) * (1 + 1e-6)
            A = (
                torch.sum(R * S_inmask.unsqueeze(0), dim=(1, 2))
                / torch.sum(S_inmask * R.shape[0])
                - torch.sum(R * S_outmask.unsqueeze(0), dim=(1, 2))
                / torch.sum(S_outmask * R.shape[0])
            ) / torch.mean(R)
            A = torch.where(torch.isnan(A), 0, A)

            # temp1 = torch.sum(p['A1_coeff'][validA1coeff_idx]**2)
            B_sq_sum = torch.sum(B**2 * B_valid)
            B_sum = torch.sum(B * B_valid)
            A_sum = torch.sum(A * B_valid)
            BA_sum = torch.sum(B * A * B_valid)
            denominator = B_sum * A_sum - BA_sum * torch.sum(B_valid)
            # limit B1_offset: 0<B1_offset<B1_uplimit

            D_Z = torch.clamp(
                (B_sq_sum * A_sum - B_sum * BA_sum) / (denominator + 1e-6),
                0,
                self.D_Z_max / torch.mean(S),
            )

            Z = D_Z * (torch.mean(S) - S)

            D_R = (R * B_valid.view(-1, 1, 1)).sum(dim=0) / B_valid.sum() - (
                B * B_valid
            ).sum() / B_valid.sum() * S
            D_R = D_R - torch.mean(D_R) - Z

            # smooth A_offset
            D_R = dct2d(D_R)
            D_R = _jshrinkage(D_R, self.smoothness_darkfield / (10.0 * mu))
            D_R = idct2d(D_R)
            D_R = _jshrinkage(D_R, self.sparse_cost_darkfield / (10.0 * mu))
            D_R = D_R + Z
        fit_residual = R - I_B
        Y = Y + mu * fit_residual
        mu = torch.minimum(mu * self.rho, torch.tensor(self.max_mu, dtype=torch.float32, device=S.device))

        # put the variables back to 4-dim input array
        S = S.unsqueeze(0)
        D_R = D_R.unsqueeze(0)
        I_R = I_R.unsqueeze(1)
        Y = Y.unsqueeze(1)
        fit_residual = fit_residual.unsqueeze(1)
        return (k + 1, S, D_R, D_Z, I_R, B, Y, mu, fit_residual, 0.0)

    def _step_only_baseline(self, Im, weight, S, D, vals):
        k, I_R, B, Y, mu, fit_residual, value_diff = vals
        Im = Im[:, 0, ...]
        weight = weight[:, 0, ...]
        S = S[0]
        D = D[0]
        I_R = I_R[:, 0, ...]
        Y = Y[:, 0, ...]

        I_B = S.unsqueeze(0) * B.view(-1, 1, 1) + D.unsqueeze(0)

        # update I_R using approximated l0 norm
        I_R = I_R + (Im - I_B - I_R + (1 / mu) * Y) / 1.0
        I_R = _jshrinkage(I_R, weight / (1.0 * mu))

        R1 = Im - I_R
        # A1_coeff = mean(R1)-mean(A_offset);
        B = torch.mean(R1, dim=(1, 2)) - torch.mean(D)
        # A1_coeff(A1_coeff<0) = 0;
        B = torch.maximum(B, torch.tensor(0.0, device=B.device))
        # Z1 = D - A1_hat - E1_hat;
        fit_residual = Im - I_B - I_R
        # Y1 = Y1 + mu*Z1;
        Y = Y + mu * fit_residual
        mu = torch.minimum(mu * self.rho, torch.tensor(self.max_mu, dtype=torch.float32, device=I_R.device))

        I_R = I_R.unsqueeze(1)
        Y = Y.unsqueeze(1)
        fit_residual = fit_residual.unsqueeze(1)
        return (k + 1, I_R, B, Y, mu, fit_residual, 0.0)

    def calc_weights(self, I_B, I_R):
        I_B = I_B[:, 0, ...]
        I_R = I_R[:, 0, ...]
        XE_norm = I_R / (torch.mean(I_B, dim=(1, 2)).view(-1, 1, 1) + 1e-6)
        weight = torch.ones_like(I_R) / (torch.abs(XE_norm) + self.epsilon)
        weight = weight / torch.mean(weight)
        return weight.unsqueeze(1)

    def calc_dark_weights(self, D_R):
        return torch.ones_like(D_R)

    def calc_weights_baseline(self, I_B, I_R):
        I_B = I_B[:, 0, ...]
        I_R = I_R[:, 0, ...]
        mean_vec = torch.mean(I_B, dim=(1, 2))
        XE_norm = mean_vec.view(-1, 1, 1) / (I_R + 1e-6)
        weight = 1.0 / (torch.abs(XE_norm) + self.epsilon)
        weight = weight / torch.mean(weight)
        return weight.unsqueeze(1)

    def calc_darkfield(_self, S, D_R, D_Z):
        return D_R + D_Z * (1 + S)
