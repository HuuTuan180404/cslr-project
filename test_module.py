import torch

from models.text_branch import TextBranch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. Fake gloss vocabulary
# ============================================================

gloss_vocab = {
    "<PAD>": 0,
    "HELLO": 1,
    "THANK": 2,
    "YOU": 3,
    "I": 4,
    "LOVE": 5,
    "GOOD": 6,
    "MORNING": 7,
}


# ============================================================
# 2. Fake gloss sequences
# ============================================================

glosses = [
    ["HELLO"],
    ["THANK", "YOU"],
    ["I", "LOVE", "YOU"],
    ["GOOD", "MORNING"],
]


# ============================================================
# 3. Convert gloss → token IDs
# ============================================================

token_sequences = [[gloss_vocab[g] for g in sequence] for sequence in glosses]

print("Token sequences:")
print(token_sequences)


# ============================================================
# 4. Padding
# ============================================================

max_len = max(len(sequence) for sequence in token_sequences)

input_ids = []

for sequence in token_sequences:
    padded = sequence + [gloss_vocab["<PAD>"]] * (max_len - len(sequence))

    input_ids.append(padded)


input_ids = torch.tensor(input_ids, device=device)

# 1 = real token
# 0 = padding
attention_mask = (input_ids != gloss_vocab["<PAD>"]).long()


# ============================================================
# 5. Create model
# ============================================================

model = TextBranch(
    vocab_size=len(gloss_vocab),
    embed_dim=256,
    depth=2,
    proj_dim=128,
    state_dim=16,
    conv_kernel=4,
    expand=2,
    dropout=0.1,
    padding_idx=gloss_vocab["<PAD>"],
).to(device)


# ============================================================
# 6. Forward
# ============================================================
outputs = model(input_ids=input_ids, attention_mask=attention_mask)


# ============================================================
# 7. Print
# ============================================================

print("\nInput:")
print(input_ids)

print("\nAttention mask:")
print(attention_mask)

print("\nShapes:")
print("token_features   :", outputs["token_features"].shape)
print("sentence_features:", outputs["sentence_features"].shape)
print("z_text            :", outputs["z_text"].shape)
