import math

import torch


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
