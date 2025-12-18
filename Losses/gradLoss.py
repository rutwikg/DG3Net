# Author - Rutwik Gulakala - gulakala@iam.rwth-aachen.de
# Grad-loss for physics-driven training from DG3Net Paper

# -------------------------
# Loss / strain utilities
# -------------------------
def compute_strain(disp, pos, edge_index):
    """
    disp: (N_nodes_kept, 3)
    pos: (N_nodes_kept, 3)
    edge_index: (2, N_edges_kept) with indices referencing the masked node array (0..N_kept-1)
    """
    if edge_index.numel() == 0:
        return torch.zeros(0, device=disp.device)

    src = edge_index[0]
    dst = edge_index[1]

    disp_src = disp[src, :]  # (N_edges, 3)
    disp_dst = disp[dst, :]  # (N_edges, 3)

    pos_src = pos[src, :]    # (N_edges, 3)
    pos_dst = pos[dst, :]    # (N_edges, 3)

    disp_diff = disp_dst - disp_src
    pos_diff = pos_dst - pos_src

    # Avoid exact zero division - small epsilon per-component
    grad_u_edge = disp_diff / (pos_diff + 1e-6)

    strain_scalar = torch.norm(grad_u_edge, dim=1)  # (N_edges,)
    return strain_scalar

eps=1e-12

def compute_lambda(pred_out, true_out, node_positions, edge_index, criterion, offset=1.0):
    """
    Compute adaptive lambda as in your snippet. Returns a scalar tensor.
    """
    print("offset=",offset)
    # displacement loss
    l_disp = criterion(pred_out, true_out)

    # strain losses (if edges exist)
    pred_strain_t = compute_strain(pred_out[:, :3], node_positions, edge_index)
    true_strain_t = compute_strain(true_out[:, :3], node_positions, edge_index)

    if pred_strain_t.numel() == 0 or true_strain_t.numel() == 0:
        l_strain = torch.tensor(eps, device=pred_out.device)
    else:
        l_strain = criterion(pred_strain_t, true_strain_t)

    exp_disp   = torch.floor(torch.log10(l_disp + eps))
    exp_strain = torch.floor(torch.log10(l_strain + eps))

    print(exp_disp, exp_strain)
    lam = torch.pow(torch.tensor(10.0, device=pred_out.device), exp_disp - exp_strain + offset)
    lam = torch.clamp(lam, 1e-16, 1e10)
    #print(lam)
    return lam


def total_loss(pred_out, true_out, node_positions, edge_index, criterion, lambda_strain=1e-6):
    """
    pred_out, true_out: (N_nodes_kept, Features)
    node_positions: (N_nodes_kept, 3)
    edge_index: (2, N_edges_kept)
    Returns: (total, l_disp, l_strain) where total is a scalar tensor
    """

    # If there are zero nodes (empty tensors), return zeros
    if pred_out.numel() == 0 or true_out.numel() == 0:
        zero = torch.tensor(0.0, device=node_positions.device, requires_grad=True)
        return zero

    l_disp = criterion(pred_out, true_out)

    # strain computed only if there are edges
    if edge_index.numel() == 0:
        l_strain = torch.tensor(0.0, device=pred_out.device)
    else:
        pred_strain_t = compute_strain(pred_out[:, :3], node_positions, edge_index)
        true_strain_t = compute_strain(true_out[:, :3], node_positions, edge_index)

        # if strain arrays are empty (no edges) guard:
        if pred_strain_t.numel() == 0 or true_strain_t.numel() == 0:
            l_strain = torch.tensor(0.0, device=pred_out.device)
        else:
            l_strain = criterion(pred_strain_t, true_strain_t)

    total = l_disp + lambda_strain * l_strain
    return total
