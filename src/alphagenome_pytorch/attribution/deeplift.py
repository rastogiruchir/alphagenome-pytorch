import math

import numpy as np
import torch


def local_dinucleotide_shuffle(
    X: torch.Tensor,
    n: int = 20,
    bin_size: int = 1024,
    min_bin_size: int = 512,
    random_state: int | None = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Dinucleotide-shuffle sequences independently within local bins.

    This function largely has the same interface as
    :func:`tangermeme.ersatz.dinucleotide_shuffle`, but each bin is shuffled
    independently and then stitched back into the full sequence.
    """
    from tangermeme.ersatz import dinucleotide_shuffle

    if X.ndim != 3 or X.shape[1] != 4:
        raise ValueError("Expected input shape (batch_size, 4, seq_length)")
    if bin_size >= X.shape[-1]:
        raise ValueError(
            "Sequence length must be longer than bin_size. "
            "Use dinucleotide_shuffle directly for shorter sequences."
        )
    if min_bin_size > bin_size:
        raise ValueError("min_bin_size must be <= bin_size")

    rng = np.random.RandomState(random_state)

    X_shuf = X.unsqueeze(1).repeat(1, n, 1, 1)
    for i in range(X.shape[0]):
        for j in range(n):
            first_cut = rng.randint(min_bin_size, bin_size + 1)
            boundaries = [0, first_cut]
            boundaries.extend(range(first_cut + bin_size, X.shape[-1], bin_size))
            boundaries.append(X.shape[-1])

            bins = list(zip(boundaries[:-1], boundaries[1:]))
            if bins[-1][1] - bins[-1][0] < min_bin_size:
                # merge last two bins if the final bin is too small
                bins[-2] = (bins[-2][0], bins[-1][1])
                bins.pop()

            for (bin_start, bin_end) in bins:
                shuffled = dinucleotide_shuffle(
                    X[i:i+1, :, bin_start:bin_end],
                    n=1,
                    random_state=rng.randint(0, 2**31 - 1),
                    verbose=verbose,
                )
                X_shuf[i, j, :, bin_start:bin_end] = shuffled[0, 0]

    return X_shuf


def softmax_symmetric_product_rule(module, grad_input, grad_output):
    """DeepLIFT rule for softmax that respects full input-output dependencies and uses
    a symmetric product rule.

    Softmax decomposition:
        a_i = exp(x_i - c)              c is a constant for numerical stability
        d = sum_i a_i
        r = 1 / d
        y_k = a_k * r
    """
    dim = module.dim
    if dim < 0:
        dim = module.input.ndim + dim
    if dim != module.input.ndim - 1:
        raise ValueError("Only softmax over the last dimension is currently supported.")

    x, x_ref = module.input.chunk(2, dim=0)
    gout, gout_ref = grad_output[0].chunk(2, dim=0)

    c = torch.maximum(
        x.max(dim=-1, keepdim=True).values,
        x_ref.max(dim=-1, keepdim=True).values,
    )
    a = torch.exp(x - c)
    a_ref = torch.exp(x_ref - c)
    d = a.sum(dim=-1, keepdim=True)
    d_ref = a_ref.sum(dim=-1, keepdim=True)
    r = d.reciprocal()
    r_ref = d_ref.reciprocal()

    # Multiplier for x_i -> exp(x_i - c), with derivative fallback.
    delta_x = x - x_ref
    delta_a = a - a_ref
    mult_x_to_a = torch.where(delta_x.abs() > 1e-6, delta_a / delta_x, a_ref)

    # Softmax has a diagonal numerator path and dense denominator path.
    #     m_{a_i -> y_k} =
    #       1{i = k} * (r + r_ref) / 2
    #       - (a_k + a_ref_k) / (2 * d * d_ref)
    midpoint_r = (r + r_ref) / 2
    midpoint_a = (a + a_ref) / 2
    reciprocal_mult = -1.0 / (d * d_ref)

    # Contract over output index k without materializing M[k, i].
    #     gin_i = m_{x_i -> a_i} * [
    #         gout_i * (r + r_ref) / 2
    #         - sum_k gout_k * (a_k + a_ref_k) / (2 * d * d_ref)
    #     ]
    gin = mult_x_to_a * (
        gout * midpoint_r
        + (gout * midpoint_a * reciprocal_mult).sum(dim=-1, keepdim=True)
    )
    gin_ref = mult_x_to_a * (
        gout_ref * midpoint_r
        + (gout_ref * midpoint_a * reciprocal_mult).sum(dim=-1, keepdim=True)
    )

    return (torch.cat([gin, gin_ref], dim=0),)


def softmax_log_ratio_product_rule(module, grad_input, grad_output):
    """DeepLIFT rule for softmax that respects full input-output dependencies and uses
    a log-ratio product rule.

    Softmax decomposition:
        a_i = exp(x_i - c)              c is a constant for numerical stability
        d = sum_i a_i
        r = 1 / d
        y_k = a_k * r
    """
    dim = module.dim
    if dim < 0:
        dim = module.input.ndim + dim
    if dim != module.input.ndim - 1:
        raise ValueError("Only softmax over the last dimension is currently supported.")

    x, x_ref = module.input.chunk(2, dim=0)
    gout, gout_ref = grad_output[0].chunk(2, dim=0)

    c = torch.maximum(
        x.max(dim=-1, keepdim=True).values,
        x_ref.max(dim=-1, keepdim=True).values,
    )
    a = torch.exp(x - c)
    a_ref = torch.exp(x_ref - c)
    d = a.sum(dim=-1, keepdim=True)
    d_ref = a_ref.sum(dim=-1, keepdim=True)
    r = d.reciprocal()
    r_ref = d_ref.reciprocal()
    y = a * r
    y_ref = a_ref * r_ref

    # Multiplier for x_i -> exp(x_i - c), with derivative fallback.
    delta_x = x - x_ref
    delta_a = a - a_ref
    mult_x_to_a = torch.where(delta_x.abs() > 1e-6, delta_a / delta_x, a_ref)

    # Log-ratio multipliers for the direct numerator path and denominator path.
    #     m_{a_i -> y_k} =
    #       1{i = k} * (Δy_k / Δlog(y_k)) * (Δlog(a_k) / Δa_k)
    #       - (1 / (d * d_ref)) * (Δy_k / Δlog(y_k)) * (Δlog(r) / Δr)
    delta_log_a = x - x_ref
    delta_log_r = torch.log(r) - torch.log(r_ref)
    delta_log_y = delta_log_a + delta_log_r
    delta_y = y - y_ref
    delta_r = r - r_ref

    delta_y_over_delta_log_y = torch.where(
        delta_log_y.abs() > 1e-6,
        delta_y / delta_log_y,
        y_ref,
    )
    delta_log_a_over_delta_a = torch.where(
        delta_a.abs() > 1e-6,
        delta_log_a / delta_a,
        a_ref.reciprocal(),
    )
    delta_log_r_over_delta_r = torch.where(
        delta_r.abs() > 1e-6,
        delta_log_r / delta_r,
        r_ref.reciprocal(),
    )

    mult_a_to_y = delta_y_over_delta_log_y * delta_log_a_over_delta_a
    mult_r_to_y = delta_y_over_delta_log_y * delta_log_r_over_delta_r
    reciprocal_mult = -1.0 / (d * d_ref)

    # Combine the direct a_i -> y_i contribution with the dense r -> y_k path.
    #     gin_i = m_{x_i -> a_i} * [
    #         gout_i * (Δy_i / Δlog(y_i)) * (Δlog(a_i) / Δa_i)
    #         - sum_k gout_k
    #             * (1 / (d * d_ref))
    #             * (Δy_k / Δlog(y_k))
    #             * (Δlog(r) / Δr)
    #     ]
    gin = mult_x_to_a * (
        gout * mult_a_to_y
        + (gout * mult_r_to_y * reciprocal_mult).sum(dim=-1, keepdim=True)
    )
    gin_ref = mult_x_to_a * (
        gout_ref * mult_a_to_y
        + (gout_ref * mult_r_to_y * reciprocal_mult).sum(dim=-1, keepdim=True)
    )

    return (torch.cat([gin, gin_ref], dim=0),)


def layernorm_symmetric_product_rule(module, grad_input, grad_output):
    """DeepLIFT rule for LayerNorm / RMSNorm using full input-output dependencies.

    LayerNorm decomposition:
        mu = mean(x)
        a_i = x_i - mu
        var = mean(a_i^2)
        s = 1 / sqrt(var + eps)
        z_i = a_i * s
        y_i = gamma * z_i + beta

    RMSNorm decomposition:
        a_i = x_i
        var = mean(a_i^2)
        s = 1 / sqrt(var + eps)
        z_i = a_i * s
        y_i = gamma * z_i + beta

    The product z_i = a_i * s is decomposed using a symmetric product rule.
    """
    x, x_ref = module.input.chunk(2)
    gout, gout_ref = grad_output[0].chunk(2)
    input_dtype = x.dtype

    normalized_shape = tuple(module.normalized_shape)
    norm_ndims = len(normalized_shape)
    N = math.prod(normalized_shape)

    # Flatten normalized dimensions: (..., N)
    leading_shape = x.shape[:-norm_ndims]
    x = x.reshape(*leading_shape, N)
    x_ref = x_ref.reshape(*leading_shape, N)
    gout = gout.reshape(*leading_shape, N)
    gout_ref = gout_ref.reshape(*leading_shape, N)

    # Centering: a_i = x_i - mean(x) for LayerNorm, a_i = x_i for RMSNorm.
    if module.rms_norm:
        a = x
        a_ref = x_ref
    else:
        a = x - x.mean(dim=-1, keepdim=True).to(dtype=input_dtype)
        a_ref = x_ref - x_ref.mean(dim=-1, keepdim=True).to(dtype=input_dtype)

    # Scale: s = 1 / sqrt(var + eps)
    #
    # Match the module forward pass: compute variance in float32,
    # then cast back to the input dtype.
    var = (a.float() ** 2).mean(dim=-1, keepdim=True)
    var_ref = (a_ref.float() ** 2).mean(dim=-1, keepdim=True)
    s = torch.rsqrt(var + module.eps).to(dtype=x.dtype)
    s_ref = torch.rsqrt(var_ref + module.eps).to(dtype=x.dtype)

    # Symmetric product factors for z_k = a_k * s.
    A = a + a_ref
    mult_a_to_z = 0.5 * (s + s_ref)
    mult_s_to_z = 0.5 * A

    # Multiplier for x_i to s.
    mult_x_to_s = (-A / N) * (s.square() * s_ref.square()) / (s + s_ref)

    # Fold affine weight into upstream multiplier.
    if module.elementwise_affine and module.weight is not None:
        gamma = module.weight.reshape(N).to(device=x.device, dtype=x.dtype)
        gamma_broadcast_shape = [1] * len(leading_shape) + [N]
        gout = gout * gamma.view(*gamma_broadcast_shape)
        gout_ref = gout_ref * gamma.view(*gamma_broadcast_shape)

    # Multiplier from x_i to z_k:
    #
    #   M_{x_i -> z_k} =
    #     = m_{x_i -> a_k} * m_{a_k -> z_k}         direct path
    #       + m_{x_i -> s} * m_{s -> z_k}           scale path
    #
    # Direct path:
    #   RMSNorm:   gout_i * (s + s_ref) / 2
    #   LayerNorm: (gout_i - mean_k gout_k) * (s + s_ref) / 2
    #
    # Scale path:
    #   m_{x_i -> s} * sum_k gout_k * (a_k + a_ref_k) / 2

    if module.rms_norm:
        direct = gout * mult_a_to_z
        direct_ref = gout_ref * mult_a_to_z
    else:
        direct = (gout - gout.mean(dim=-1, keepdim=True)) * mult_a_to_z
        direct_ref = (gout_ref - gout_ref.mean(dim=-1, keepdim=True)) * mult_a_to_z

    scale = mult_x_to_s * (gout * mult_s_to_z).sum(dim=-1, keepdim=True)
    scale_ref = mult_x_to_s * (gout_ref * mult_s_to_z).sum(dim=-1, keepdim=True)

    gin = direct + scale
    gin_ref = direct_ref + scale_ref

    # Restore original shapes.
    gin = gin.reshape(*leading_shape, *normalized_shape)
    gin_ref = gin_ref.reshape(*leading_shape, *normalized_shape)

    # Mixed precision LayerNorm can have bf16 input and fp32 output; hooks must
    # return input dtype.
    if grad_input[0] is not None:
        gin = gin.to(dtype=grad_input[0].dtype)
        gin_ref = gin_ref.to(dtype=grad_input[0].dtype)

    return (torch.cat([gin, gin_ref], dim=0),)


def layernorm_log_ratio_product_rule(module, grad_input, grad_output):
    """DeepLIFT rule for LayerNorm / RMSNorm using full input-output dependencies.

    LayerNorm:
        a_i = x_i - mean(x)
        var = mean_i a_i^2
        s = 1 / sqrt(var + eps)
        z_i = a_i * s
        y_i = gamma_i * z_i + beta_i

    RMSNorm:
        a_i = x_i
        var = mean_i a_i^2
        s = 1 / sqrt(var + eps)
        z_i = a_i * s
        y_i = gamma_i * z_i + beta_i

    The product z_i = a_i * s is handled with a signed log-ratio product rule,
    where s is strictly positive but a_i may be positive or negative.
    """
    x, x_ref = module.input.chunk(2)
    gout, gout_ref = grad_output[0].chunk(2)
    input_dtype = x.dtype

    normalized_shape = tuple(module.normalized_shape)
    norm_ndims = len(normalized_shape)
    N = math.prod(normalized_shape)

    # Flatten normalized dimensions: (..., N)
    leading_shape = x.shape[:-norm_ndims]
    x = x.reshape(*leading_shape, N)
    x_ref = x_ref.reshape(*leading_shape, N)
    gout = gout.reshape(*leading_shape, N)
    gout_ref = gout_ref.reshape(*leading_shape, N)

    # Centering: a_i = x_i - mean(x) for LayerNorm, a_i = x_i for RMSNorm.
    if module.rms_norm:
        a = x
        a_ref = x_ref
    else:
        a = x - x.mean(dim=-1, keepdim=True).to(dtype=input_dtype)
        a_ref = x_ref - x_ref.mean(dim=-1, keepdim=True).to(dtype=input_dtype)

    # Scale: s = 1 / sqrt(var + eps)
    #
    # Match the module forward pass: compute variance in float32,
    # then cast back to the input dtype.
    var = (a.float() ** 2).mean(dim=-1, keepdim=True)
    var_ref = (a_ref.float() ** 2).mean(dim=-1, keepdim=True)
    s = torch.rsqrt(var + module.eps).to(dtype=x.dtype)
    s_ref = torch.rsqrt(var_ref + module.eps).to(dtype=x.dtype)

    # Normalized activations: z_i = a_i * s
    z = a * s
    z_ref = a_ref * s_ref

    # Multiplier for x_i to s.
    A = a + a_ref
    mult_x_to_s = (-A / N) * (s.square() * s_ref.square()) / (s + s_ref)

    # Product-rule multipliers for z_k = a_k * s, where s > 0.
    #
    # If a_k keeps the same sign as a_ref_k, use the signed log-ratio rule:
    #
    #   m_{a_k -> z_k}
    #       = (Δz_k / Δlog|z_k|) * (Δlog|a_k| / Δa_k)
    #
    #   m_{s -> z_k}
    #       = (Δz_k / Δlog|z_k|) * (Δlog(s) / Δs)
    #
    # If a_k changes sign, assign the product contribution entirely to a_k:
    #
    #   m_{a_k -> z_k} = Δz_k / Δa_k
    #   m_{s -> z_k} = 0
    #
    delta_a = a - a_ref
    delta_s = s - s_ref
    delta_z = z - z_ref

    same_sign = ((a > 0) & (a_ref > 0)) | ((a < 0) & (a_ref < 0))

    delta_log_abs_a = torch.log(a.abs()) - torch.log(a_ref.abs())
    delta_log_s = torch.log(s) - torch.log(s_ref)
    delta_log_abs_z = delta_log_abs_a + delta_log_s

    mult_a_to_z_same_sign = (
        torch.where(
            delta_log_abs_z.abs() > 1e-6,
            delta_z / delta_log_abs_z,
            z_ref,
        )
        * torch.where(
            delta_a.abs() > 1e-6,
            delta_log_abs_a / delta_a,
            a_ref.reciprocal(),
        )
    )
    mult_a_to_z_sign_change = torch.where(
        delta_a.abs() > 1e-6,
        delta_z / delta_a,
        s_ref,
    )
    mult_a_to_z = torch.where(
        same_sign,
        mult_a_to_z_same_sign,
        mult_a_to_z_sign_change,
    )

    mult_s_to_z_same_sign = (
        torch.where(
            delta_log_abs_z.abs() > 1e-6,
            delta_z / delta_log_abs_z,
            z_ref,
        )
        * torch.where(
            delta_s.abs() > 1e-6,
            delta_log_s / delta_s,
            s_ref.reciprocal(),
        )
    )
    mult_s_to_z = torch.where(
        same_sign,
        mult_s_to_z_same_sign,
        torch.zeros_like(mult_s_to_z_same_sign),
    )

    # Fold affine weight into the upstream multiplier.
    if module.elementwise_affine and module.weight is not None:
        gamma = module.weight.reshape(N).to(device=x.device, dtype=x.dtype)
        gamma_broadcast_shape = [1] * len(leading_shape) + [N]
        gout = gout * gamma.view(*gamma_broadcast_shape)
        gout_ref = gout_ref * gamma.view(*gamma_broadcast_shape)

    # Direct path:
    #   RMSNorm direct_i = gout_i * m_{a_i -> z_i}
    #
    #   LayerNorm direct_i =
    #       gout_i * m_{a_i -> z_i}
    #       - mean_k [gout_k * m_{a_k -> z_k}]
    #
    # Scale path:
    #   m_{x_i -> s} * sum_k gout_k * m_{s -> z_k}
    if module.rms_norm:
        direct = gout * mult_a_to_z
        direct_ref = gout_ref * mult_a_to_z
    else:
        direct = (
            gout * mult_a_to_z
            - (gout * mult_a_to_z).mean(dim=-1, keepdim=True)
        )
        direct_ref = (
            gout_ref * mult_a_to_z
            - (gout_ref * mult_a_to_z).mean(dim=-1, keepdim=True)
        )

    scale = mult_x_to_s * (gout * mult_s_to_z).sum(dim=-1, keepdim=True)
    scale_ref = mult_x_to_s * (gout_ref * mult_s_to_z).sum(dim=-1, keepdim=True)

    gin = direct + scale
    gin_ref = direct_ref + scale_ref

    # Restore original shape.
    gin = gin.reshape(*leading_shape, *normalized_shape)
    gin_ref = gin_ref.reshape(*leading_shape, *normalized_shape)

    # Hooks must return the input dtype.
    if grad_input[0] is not None:
        gin = gin.to(dtype=grad_input[0].dtype)
        gin_ref = gin_ref.to(dtype=grad_input[0].dtype)

    return (torch.cat([gin, gin_ref], dim=0),)


def bilinear_symmetric_product_rule(module, grad_input, grad_output):
    """DeepLIFT symmetric product rule for bilinear operations."""
    left, left_ref = module.left.chunk(2)
    right, right_ref = module.right.chunk(2)
    gout, gout_ref = grad_output[0].chunk(2)

    left_mid = (0.5 * (left + left_ref)).detach().requires_grad_(True)
    right_mid = (0.5 * (right + right_ref)).detach().requires_grad_(True)

    with torch.enable_grad():
        if module.equation is None:
            out = torch.matmul(left_mid, right_mid)
        else:
            out = torch.einsum(module.equation, left_mid, right_mid)

        gin_left, gin_right = torch.autograd.grad(
            out,
            (left_mid, right_mid),
            grad_outputs=gout,
            retain_graph=True,
            create_graph=False,
        )

        gin_left_ref, gin_right_ref = torch.autograd.grad(
            out,
            (left_mid, right_mid),
            grad_outputs=gout_ref,
            retain_graph=False,
            create_graph=False,
        )

    return (
        torch.cat([gin_left, gin_left_ref], dim=0),
        torch.cat([gin_right, gin_right_ref], dim=0),
    )
