from typing import Literal

import torch

from operator_learning.nn.linalg import covariance, sqrtmh
from operator_learning.structs import EigResult, FitResult


def vamp_score(X, Y, schatten_norm: int = 2, center_covariances: bool = True):
    """Variational Approach for learning Markov Processes (VAMP) score by :footcite:t:`Wu2019`.

    Args:
        X (torch.Tensor): Covariates for the initial time steps.
        Y (torch.Tensor): Covariates for the evolved time steps.
        schatten_norm (int, optional): Computes the VAMP-p score with ``p = schatten_norm``. Defaults to 2.
        center_covariances (bool, optional): Use centered covariances to compute the VAMP score. Defaults to True.

    Raises:
        NotImplementedError: If ``schatten_norm`` is not 1 or 2.

    Returns:
        torch.Tensor: VAMP score
    """
    cov_X, cov_Y, cov_XY = (
        covariance(X, center=center_covariances),
        covariance(Y, center=center_covariances),
        covariance(X, Y, center=center_covariances),
    )
    if schatten_norm == 2:
        # Using least squares in place of pinv for numerical stability
        M_X = torch.linalg.lstsq(cov_X, cov_XY).solution
        M_Y = torch.linalg.lstsq(cov_Y, cov_XY.T).solution
        return torch.trace(M_X @ M_Y)
    elif schatten_norm == 1:
        sqrt_cov_X = sqrtmh(cov_X)
        sqrt_cov_Y = sqrtmh(cov_Y)
        M = torch.linalg.multi_dot(
            [
                torch.linalg.pinv(sqrt_cov_X, hermitian=True),
                cov_XY,
                torch.linalg.pinv(sqrt_cov_Y, hermitian=True),
            ]
        )
        return torch.linalg.matrix_norm(M, "nuc")
    else:
        raise NotImplementedError(f"Schatten norm {schatten_norm} not implemented")


def dp_score(
    X: torch.Tensor,
    Y: torch.Tensor,
    relaxed: bool = True,
    metric_deformation: float = 1.0,
    center_covariances: bool = True,
):
    """Deep Projection score by :footcite:t:`Kostic2023DPNets`.

    Args:
        X (torch.Tensor): Covariates for the initial time steps.
        Y (torch.Tensor): Covariates for the evolved time steps.
        relaxed (bool, optional): Whether to use the relaxed (more numerically stable) or the full deep-projection loss. Defaults to True.
        metric_deformation (float, optional): Strength of the metric metric deformation loss: Defaults to 1.0.
        center_covariances (bool, optional): Use centered covariances to compute the Deep Projection score. Defaults to True.

    Returns:
        torch.Tensor: Deep Projection score
    """
    cov_X, cov_Y, cov_XY = (
        covariance(X, center=center_covariances),
        covariance(Y, center=center_covariances),
        covariance(X, Y, center=center_covariances),
    )
    R_X = logfro_loss(cov_X)
    R_Y = logfro_loss(cov_Y)
    if relaxed:
        S = (torch.linalg.matrix_norm(cov_XY, ord="fro") ** 2) / (
            torch.linalg.matrix_norm(cov_X, ord=2)
            * torch.linalg.matrix_norm(cov_Y, ord=2)
        )
    else:
        M_X = torch.linalg.lstsq(cov_X, cov_XY).solution
        M_Y = torch.linalg.lstsq(cov_Y, cov_XY.T).solution
        S = torch.trace(M_X @ M_Y)
    return S - 0.5 * metric_deformation * (R_X + R_Y)


def spectral_contrastive_score(X: torch.Tensor, Y: torch.Tensor):
    # Credits https://github.com/jhaochenz96/spectral_contrastive_learning/blob/ee431bdba9bb62ad00a7e55792213ee37712784c/models/spectral.py#L8C1-L17C96
    assert X.shape == Y.shape
    assert X.ndim == 2

    npts, dim = X.shape
    diag = 2 * torch.mean(X * Y) * dim
    square_term = torch.matmul(X, Y.T) ** 2
    off_diag = -(
        torch.mean(
            torch.triu(square_term, diagonal=1) + torch.tril(square_term, diagonal=-1)
        )
        * npts
        / (npts - 1)
    )
    return diag + off_diag


def logfro_loss(cov: torch.Tensor):
    """Logarithmic + Frobenious (metric deformation) loss as used in :footcite:t:`Kostic2023DPNets`, defined as :math:`{{\\rm Tr}}(C^{2} - C -\ln(C))` .

    Args:
        cov (torch.tensor): A symmetric positive-definite matrix.

    Returns:
        torch.tensor: Loss function
    """
    eps = torch.finfo(cov.dtype).eps * cov.shape[0]
    vals_x = torch.linalg.eigvalsh(cov)
    vals_x = torch.where(vals_x > eps, vals_x, eps)
    loss = torch.mean(-torch.log(vals_x) + vals_x * (vals_x - 1.0))
    return loss


def fit_ridgels(
    cov_X: torch.Tensor,
    tikhonov_reg: float = 0.0,
) -> FitResult:
    """Fit the ridge least squares estimator for the transfer operator.

    Args:
        cov_X (torch.Tensor): covariance matrix of the input data.
        tikhonov_reg (float, optional): Ridge regularization. Defaults to 0.0.

    Returns:
        FitResult: as defined in operator_learning.structs
    """
    dim = cov_X.shape[0]
    reg_input_covariance = cov_X + tikhonov_reg * torch.eye(
        dim, dtype=cov_X.dtype, device=cov_X.device
    )
    values, vectors = torch.linalg.eigh(reg_input_covariance)
    # Divide columns of vectors by square root of eigenvalues
    rsqrt_evals = 1.0 / torch.sqrt(values + 1e-10)
    Q = vectors @ torch.diag(rsqrt_evals)
    result: FitResult = FitResult({"U": Q, "V": Q, "svals": values})
    return result


def eig(
    fit_result: FitResult,
    cov_XY: torch.Tensor,
) -> EigResult:
    """Computes the eigendecomposition of the transfer operator. It only computes the right eigenvectors.

    Args:
        fit_result (FitResult): Fit result from the fit_ridgels function.
        cov_XY (torch.Tensor): Cross covariance matrix between the input and output data.

    Returns:
        EigResult: as defined in operator_learning.structs
    """
    U = fit_result["U"]
    # Using the trick described in https://arxiv.org/abs/1905.11490
    M = torch.linalg.multi_dot([U.T, cov_XY, U])
    values, rv = torch.linalg.eig(M)

    # Normalization in RKHS norm
    rv = U.cfloat() @ rv
    rv = rv / torch.linalg.norm(rv, axis=0)
    # # Biorthogonalization
    # lv = torch.linalg.multi_dot([cov_XY.T, U, lv])
    # lv = lv[:, l_perm]
    # l_norm = torch.sum(lv * rv, axis=0)
    # lv = lv / l_norm
    result: EigResult = EigResult({"values": values, "left": None, "right": rv})
    return result


def evaluate_eigenfunction(
    X: torch.Tensor,
    eig_result: EigResult,
    which: Literal["left", "right"] = "right",
):
    if which == "left":
        raise NotImplementedError("Left eigenfunctions are not implemented")
    vr_or_vl = eig_result[which]
    return X.to(vr_or_vl.dtype) @ vr_or_vl
