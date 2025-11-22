import torch
import torch.optim as optim
from torch.nn.functional import softmax

def _apply_convolve_mode(conv_result: torch.Tensor, x_length: int, y_length: int, mode: str) -> torch.Tensor:
    valid_convolve_modes = ["full", "valid", "same"]
    if mode == "full":
        return conv_result
    elif mode == "valid":
        target_length = max(x_length, y_length) - min(x_length, y_length) + 1
        start_idx = (conv_result.size(-1) - target_length) // 2
        return conv_result[..., start_idx : start_idx + target_length]
    elif mode == "same":
        start_idx = (conv_result.size(-1) - x_length) // 2
        return conv_result[..., start_idx : start_idx + x_length]
    else:
        raise ValueError(f"Unrecognized mode value '{mode}'. Please specify one of {valid_convolve_modes}.")
    

def _check_shape_compatible(x: torch.Tensor, y: torch.Tensor) -> None:
    if x.ndim != y.ndim:
        raise ValueError(f"The operands must be the same dimension (got {x.ndim} and {y.ndim}).")

    for i in range(x.ndim - 1):
        xi = x.size(i)
        yi = y.size(i)
        if xi == yi or xi == 1 or yi == 1:
            continue
        raise ValueError(f"Leading dimensions of x and y are not broadcastable (got {x.shape} and {y.shape}).")


def _check_convolve_mode(mode: str) -> None:
    valid_convolve_modes = ["full", "valid", "same"]
    if mode not in valid_convolve_modes:
        raise ValueError(f"Unrecognized mode value '{mode}'. Please specify one of {valid_convolve_modes}.")
    

def convolve(x: torch.Tensor, y: torch.Tensor, mode: str = "full") -> torch.Tensor:

    _check_shape_compatible(x, y)
    _check_convolve_mode(mode)

    x_size, y_size = x.size(-1), y.size(-1)

    if x.size(-1) < y.size(-1):
        x, y = y, x

    if x.shape[:-1] != y.shape[:-1]:
        new_shape = [max(i, j) for i, j in zip(x.shape[:-1], y.shape[:-1])]
        x = x.broadcast_to(new_shape + [x.shape[-1]])
        y = y.broadcast_to(new_shape + [y.shape[-1]])

    num_signals = torch.tensor(x.shape[:-1]).prod()
    reshaped_x = x.reshape((int(num_signals), x.size(-1)))
    reshaped_y = y.reshape((int(num_signals), y.size(-1)))
    output = torch.nn.functional.conv1d(
        input=reshaped_x.unsqueeze(1),
        weight=reshaped_y.flip(-1).unsqueeze(1),
        stride=1,
        groups=reshaped_x.size(0),
        padding=reshaped_y.size(-1) - 1,
    )
    output_shape = x.shape[:-1] + (-1,)
    result = output.reshape(output_shape)
    return _apply_convolve_mode(result, x_size, y_size, mode)


def transiton_max(ratios, M):
    ps = torch.cumsum(ratios, dim=0)
    ps_shifted = torch.cat((torch.tensor([0]), ps[:-1]))

    return torch.pow(ps, M) - torch.pow(ps_shifted, M)


def transiton_sum(ratios, M):
    out = ratios
    for _ in range(M - 1):
        out = convolve(out, ratios)
    return out


def transiton_smallest_common_parent(ratios, label_vec_to_partial):
    """
    currently, it is assumed that M = 2 
    """
    out = torch.zeros(label_vec_to_partial["total"])
    c = len(ratios)
    for i in range(c):
        for j in range(c):
            index = label_vec_to_partial[(i,j)]
            out[index] += ratios[i] * ratios[j]
    
    return out


def solver_mirror_descent(transition, partial_ratio, n_class, transition_para, n_iter=5000, lr=0.001):

    # Initialization
    log_ratios = torch.ones(n_class)
    log_ratios.cuda()
    log_ratios.requires_grad = True

    # Set up the optimizer
    optimizer = optim.Adam([log_ratios], lr=lr)

    # Optimization loop
    for _ in range(n_iter):
        optimizer.zero_grad()
        est_ratio = transition(softmax(log_ratios, dim=0), transition_para)
        loss = -torch.inner(torch.log(est_ratio), partial_ratio)
        loss.backward()
        optimizer.step()

    return softmax(log_ratios, dim=0)