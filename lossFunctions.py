import torch


<<<<<<< Updated upstream
def lossFunction(F_fixed: torch, F_wrapped: torch, loss: str = "l2") -> float:
=======
def lossFunction(F_fixed: torch, F_wrapped: torch, loss: str = "l2") -> torch:
>>>>>>> Stashed changes
    """lossFunction computes the loss value between two feature maps. 

    Args:
        F_fixed (torch): [B, C, H, W]
        F_wrapped (torch): [B, C, H, W]
        loss (str, optional): loss function type - l1, l2, Linf. Defaults to "l2".

    Raises:
        ValueError: not acceptable loss type

    Returns:
<<<<<<< Updated upstream
        float: loss value
=======
        torch: loss value
>>>>>>> Stashed changes
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


<<<<<<< Updated upstream
def L1_norm(F_fixed: torch, F_wrapped: torch) -> float:
=======
def L1_norm(F_fixed: torch, F_wrapped: torch) -> torch:
>>>>>>> Stashed changes
    """MAE

    Args:
        F_fixed (torch): [B, C, H, W]
        F_wrapped (torch): [B, C, H, W]
    """
<<<<<<< Updated upstream
    return (F_fixed - F_wrapped).abs().mean().item()


def L2_norm(F_fixed: torch, F_wrapped: torch) -> float:
=======
    return (F_fixed - F_wrapped).abs().mean()


def L2_norm(F_fixed: torch, F_wrapped: torch) -> torch:
>>>>>>> Stashed changes
    """MSE

    Args:
        F_fixed (torch): [B, C, H, W]
        F_wrapped (torch): [B, C, H, W]
    """
<<<<<<< Updated upstream
    return (F_fixed - F_wrapped).pow(2).mean().item()


def Linf_norm(F_fixed: torch, F_wrapped: torch) -> float:
=======
    return (F_fixed - F_wrapped).pow(2).mean()


def Linf_norm(F_fixed: torch, F_wrapped: torch) -> torch:
>>>>>>> Stashed changes
    """max value

    Args:
        F_fixed (torch): [B, C, H, W]
        F_wrapped (torch): [B, C, H, W]
    """
<<<<<<< Updated upstream
    return (F_fixed - F_wrapped).abs().max().item()
=======
    return (F_fixed - F_wrapped).abs().max()
>>>>>>> Stashed changes
