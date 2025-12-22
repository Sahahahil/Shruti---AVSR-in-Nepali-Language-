import torch

def ctc_decode(logits, idx2char):
    """
    logits: [T, vocab_size]
    idx2char: dictionary {index: char}
    """
    preds = torch.argmax(logits, dim=-1).tolist()  # greedy decoding
    prev = None
    chars = []
    for p in preds:
        if p != prev and p != 0:  # assuming 0 is blank
            chars.append(idx2char[p])
        prev = p
    return "".join(chars)
