# 2025-01-13

Forked, trying to get it running as a notebook with result. Trying to compare baseline vs trained. Right now it's a super small model and small number of samples, so I'm looking for a tiny diff, maybe just overfitting, before I do a full run.

Done
- Eval and inference seem broken I'll try to fix/refactor... although only when I load a checkpoint which is weird...

TODO:
- filter dataset, o 18 and double speed
- [ ] bug, when not thought present and I try to split it has an error. I changed it to return empty though tensors

# 2025-01-13 06:43:12

I'm confused by the token lenght and the fixes. So in train_forward we have

```py
print("input_ids", input_ids.shape)
print("thought_ids", thought_ids.shape)
print("language_ids", language_ids.shape)

print("attention_mask", attention_mask.shape)
print("thought_mask", thought_mask.shape)
print("language_mask? ")

print("labels", labels.shape)
print("language_labels", language_labels.shape)
print("thought_labels?")

# so Q why 256 -> 226 and 80, that doesn't add up
# the we "fix" it by adding thought_ids -1 (=1) to the label and attn

# input_ids torch.Size([18, 256])
# thought_ids torch.Size([18, 80])
# language_ids torch.Size([18, 226])

# attention_mask torch.Size([18, 256])
# thought_mask torch.Size([18, 80])
# language_mask? 

# labels torch.Size([18, 256])
# language_labels torch.Size([18, 226])
# thought_labels?
```

Q is thoughts_forward mean to output something... I'm getting nothign
