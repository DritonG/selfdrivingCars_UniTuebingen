import numpy as np
import torch
import torch.nn.functional as F


def perform_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device):
    """ Perform a deep Q-learning step
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    optimizer: torch.optim.Adam
        optimizer
    replay_buffer: ReplayBuffer
        replay memory storing transitions
    batch_size: int
        size of batch to sample from replay memory 
    gamma: float
        discount factor used in Q-learning update
    device: torch.device
        device on which to the models are allocated
    Returns
    -------
    float
        loss value for current learning step
    """

    # TODO: Run single Q-learning step
    """ Steps: 
        1. Sample transitions from replay_buffer
        2. Compute Q(s_t, a)
        3. Compute \max_a Q(s_{t+1}, a) for all next states.
        4. Mask next state values where episodes have terminated
        5. Compute the target
        6. Compute the loss
        7. Calculate the gradients
        8. Clip the gradients
        9. Optimize the model
    """
    if len(replay_buffer) < batch_size:
        return
    obses_t, action, rewards, obses_tp1, done = replay_buffer.sample(batch_size)

    states = torch.tensor(obses_t, dtype=torch.float).to(device)
    actions = torch.tensor(action, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards).to(device)
    states_ = torch.tensor(obses_tp1, dtype=torch.float).to(device)
    print("CHECK: states.shape: ", states.shape)
    print("CHECK: states_.shape: ", states_.shape)

    # compute Q(s_t, a)
    q_values = policy_net(states)[actions]
    print("CHECK: q_values.shape: ", q_values.shape)
    # Compute values for all next states
    mask = torch.tensor(tuple(map(lambda s: s is not None, states_)), device=device, dtype=torch.uint8)
    mask_next = torch.cat([s for s in states_ if s is not None])
    print("CHECK: mask_next.shape: ", mask_next.shape)

    q_next = torch.zeros(batch_size, device=device)
    print("CHECK: q_next.shape: ", q_next.shape)
    q_next[mask] = target_net(mask_next)
    print("CHECK: mask.shape: ", mask.shape)
    print("CHECK: mask_next.shape: ", mask_next.shape)


    # Compute the target
    target_q_values = (q_next * gamma) + rewards

    # compute the loss
    mse = torch.nn.MSELoss()
    # loss = F.smooth_l1_loss(q_values, target_q_values.unsqueeze(1))

    print("CHECK: rewards.shape: ", rewards.shape)
    print("CHECK: q_values.shape: ", q_values.shape)
    print("CHECK: target_q_values.shape: ", target_q_values.shape)
    loss = mse(q_values, target_q_values).to(device)

    # Optimize the model and clipping gradient
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def update_target_net(policy_net, target_net):
    """ Update the target network
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    """

    # TODO: Update target network
    return target_net.load_state_dict(policy_net.state_dict())

