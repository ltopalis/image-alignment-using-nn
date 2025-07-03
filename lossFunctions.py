import torch


def lossFunction(F_fixed: torch, F_wrapped: torch, loss: str = "l2") -> float:
    """lossFunction computes the loss value between two feature maps. 

    Args:
        F_fixed (torch): [B, C, H, W]
        F_wrapped (torch): [B, C, H, W]
        loss (str, optional): loss function type - l1, l2, Linf. Defaults to "l2".

    Raises:
        ValueError: not acceptable loss type

    Returns:
        float: loss value
    """

    loss = loss.lower()
    if loss == "l1":
        func = L1_norm
    elif loss == "l2":
        func = L2_norm
    elif loss == "linf":
        func = Linf_norm
    else:
        raise ValueError(f"Unknown loss type: {loss}")

    return func(F_fixed, F_wrapped)


def L1_norm(F_fixed: torch, F_wrapped: torch) -> float:
    """MAE

    Args:
        F_fixed (torch): [B, C, H, W]
        F_wrapped (torch): [B, C, H, W]
    """
    return (F_fixed - F_wrapped).abs().mean().item()


def L2_norm(F_fixed: torch, F_wrapped: torch) -> float:
    """MSE

    Args:
        F_fixed (torch): [B, C, H, W]
        F_wrapped (torch): [B, C, H, W]
    """
    return (F_fixed - F_wrapped).pow(2).mean().item()


def Linf_norm(F_fixed: torch, F_wrapped: torch) -> float:
    """max value

    Args:
        F_fixed (torch): [B, C, H, W]
        F_wrapped (torch): [B, C, H, W]
    """
    return (F_fixed - F_wrapped).abs().max().item()
