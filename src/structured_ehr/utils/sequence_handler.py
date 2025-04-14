import torch
from torch.nn.utils.rnn import unpad_sequence


def unpad_y(preds, labels, lens):
    raw_device = preds.device
    device = torch.device("cpu")
    preds, labels, lens = preds.squeeze(dim=-1).to(device), labels.squeeze(dim=-1).to(device), lens.to(device)
    if preds.dim() == 2:
        preds_unpad = unpad_sequence(preds, batch_first=True, lengths=lens)
        preds_unpad = [pred[-1] for pred in preds_unpad]
        preds = torch.vstack(preds_unpad)
    preds = preds.squeeze(dim=-1)
    if labels.dim() == 2:
        labels_unpad = unpad_sequence(labels, batch_first=True, lengths=lens)
        labels_unpad = [label[-1] for label in labels_unpad]
        labels = torch.vstack(labels_unpad)
    labels = labels.squeeze(dim=-1)
    return preds.to(raw_device), labels.to(raw_device)


def unpad_batch(x, y, lens):
    x = x.detach().cpu()
    y = y.detach().cpu()
    lens = lens.detach().cpu()
    x_unpad = unpad_sequence(x, batch_first=True, lengths=lens)
    x_unpad = [x[-1] for x in x_unpad]
    x_stack = torch.vstack(x_unpad).squeeze(dim=-1)
    y_unpad = unpad_sequence(y, batch_first=True, lengths=lens)
    y_unpad = [y[-1] for y in y_unpad]
    y_stack = torch.vstack(y_unpad).squeeze(dim=-1)
    return x_stack.numpy().squeeze(), y_stack.numpy().squeeze()


def generate_mask(seq_lens):
    """Generates a mask for the sequence.

    Args:
        seq_lens: [batch size]
        (max_len: int)

    Returns:
        mask: [batch size, max_len]
    """
    max_len = torch.max(seq_lens).to(seq_lens.device)
    mask = torch.arange(max_len).expand(len(seq_lens), max_len).to(seq_lens.device)
    mask = mask < seq_lens.unsqueeze(1)
    return mask


def get_last_visit(hidden_states, mask):
    """Gets the last visit from the sequence model.

    Args:
        hidden_states: [batch size, seq len, hidden_size]
        mask: [batch size, seq len]

    Returns:
        last_visit: [batch size, hidden_size]
    """
    if mask is None:
        return hidden_states[:, -1, :]
    else:
        mask = mask.long()
        last_visit = torch.sum(mask, 1) - 1
        last_visit = last_visit.unsqueeze(-1)
        last_visit = last_visit.expand(-1, hidden_states.shape[1] * hidden_states.shape[2])
        last_visit = torch.reshape(last_visit, hidden_states.shape)
        last_hidden_states = torch.gather(hidden_states, 1, last_visit)
        last_hidden_state = last_hidden_states[:, 0, :]
        return last_hidden_state