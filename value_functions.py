import torch
import torch.nn as nn
import math

def compute_VkN(states, rewards, tau, k, H, gamma, value_fn_rewards):
    """
    Compute V_N^k(s_tau) for one rollout.

    states: torch.Tensor of shape [batch_size, t, state_dim]
    rewards: torch.Tensor of shape [batch_size, t, 1]
    tau: the index at which we want the value
    k: how many steps of returns we sum before bootstrapping
    H: the horizon from the paper (for safety, we don't exceed t+H)
    gamma: discount factor (0 < gamma <= 1)
    value_fn_rewards: torch.Tensor of shape [batch_size, t, 1]

    Returns: torch.Tensor of shape [batch_size, 1] value estimate of V_N^k(s_tau)
    """

    # We'll define h = min(tau + k, tau + H), i.e., we don't exceed the horizon
    h = min(tau + k, tau + H)

    # 1) Sum discounted rewards from tau up to h-1
    discounted_return = torch.zeros(rewards.size(0), 1, device=rewards.device)
    for n in range(tau, h):
        # discount exponent is (n - tau)
        # print(f"n: {n}, tau: {tau}, gamma: {gamma}, rewards shape: {rewards.shape}, discounted_return shape: {discounted_return.shape}")
        discounted_return += (gamma ** (n - tau)) * rewards[:, n, :].detach()

    # 2) Add the bootstrap from the value function reward at index h
    #    only if h < tau + H (i.e., we haven't hit the horizon yet)
    if h < tau + H:
        discounted_return += (gamma ** (h - tau)) * value_fn_rewards[:, h, :].detach()

    return discounted_return

def compute_Vlambda(states, rewards, tau, H, gamma, lam, value_fn_rewards):
    """
    Compute V_lambda(s_tau) and return a tensor of V_lambda values for each step.

    states: torch.Tensor of shape [batch_size, t, state_dim]
    rewards: torch.Tensor of shape [batch_size, t, 1]
    tau: starting index for the value
    H: horizon for imagination
    gamma: discount factor
    lam: lambda (0 <= lam <= 1)
    value_fn_rewards: torch.Tensor of shape [batch_size, t, 1]

    Returns: torch.Tensor of shape [batch_size, H, 1] for each step
    """

    # Tensor to store V_lambda values for each step
    v_lambda_tensor = torch.zeros(rewards.size(0), H, 1, device=rewards.device)
    
    # n goes from 1 to H, but note that the formula has H-1 in the sum
    # plus a separate last term for n=H. We'll just unify them in code.
    for n in range(1, H + 1):
        # weight for the n-th return
        if n < H:
            weight = (1 - lam) * (lam ** (n - 1))
        else:
            # for n == H
            weight = lam ** (H - 1)

        # compute the n-step return V_N^n
        vkN = compute_VkN(states, rewards, tau, n, H, gamma, value_fn_rewards)
        
        # Calculate the weighted V_lambda for this step
        v_lambda_tensor[:, n-1, :] = weight * vkN

    return v_lambda_tensor
