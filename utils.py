def generate_toy_fk(batch_size, num_joints):
    import numpy as np
    import torch

    lengths = np.ones(num_joints) * 1.0  # segment lengths
    angles = np.random.uniform(-np.pi, np.pi, (batch_size, num_joints))  # joint angles
    roots = np.random.uniform(-5, 5, (batch_size, 2))  # random root positions

    rigs = np.concatenate([roots, angles], axis=-1)  # (batch_size, 2 + num_joints)

    anchors = np.zeros((batch_size, num_joints, 4))  # [x, y, cos(\theta), sin(\theta)]

    for b in range(batch_size):
        x, y = rigs[b, 0], rigs[b, 1]  # root position
        total_angle = 0.0

        for j in range(num_joints):
            total_angle += rigs[b, 2 + j]  # accumulate angles
            dx = lengths[j] * np.cos(total_angle)
            dy = lengths[j] * np.sin(total_angle)
            x += dx
            y += dy
            anchors[b, j, 0] = x
            anchors[b, j, 1] = y
            anchors[b, j, 2] = np.cos(total_angle)
            anchors[b, j, 3] = np.sin(total_angle)

    return torch.tensor(anchors, dtype=torch.float32), torch.tensor(rigs, dtype=torch.float32)