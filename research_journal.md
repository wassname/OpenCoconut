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


Ah in training I am now getting a mix between kv cache size and inputs


The expanded size of the tensor (299) must match the existing size (214) at non-singleton dimension 3.  Target sizes: [12, 14, 214, 299].  Tensor sizes: [12, 1, 214, 214]


query.shape
torch.Size([12, 14, 214, 64])
key.shape
torch.Size([12, 14, 299, 64])
value.shape
torch.Size([12, 14, 299, 64])

input_ids.shape
torch.Size([12, 256])

and 2 thoughts
thought_hidden[0].shape
torch.Size([12, 2, 896])
len(thought_hidden)
25

input_ids[:, :bot_pos+1].shape
torch.Size([12, 42])
2 thoughts

so there are either 43 or 85 missing?
which is the shape of the prefix... this is added on each forward through....



len(past_kv.key_cache)
24
past_kv.key_cache[0].shape
torch.Size([12, 2, 299, 64])

maybe just chacne cache position,
how did lucidrains do it?

can use?

    cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
        Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
        this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
        the complete sequence length.

past_key_values.get_seq_length()

should be like

    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
    )


hm so cache position doesn't help
maybe I can just append the last part of the thinking cache onto the last one???


# 2025-01-14 20:49:54

Ok so kv cache does not support multiple forwards!!! there are various issues with this, but if you go forward 40 with a cache of 30, the new cache is 70


, so we have two paths
- run two forwards, on to generate thought hidden states (self.model.forward), one to go forward
- or go on token at a time using kv cache, this is less elegent but faster as it's one pass... ok this only handles a batch of one?, soo refactor to use the top one. Otherwise we would do one token forward pass over the whole batch
- or just change it so every single forward has token embedding plus last hidden? this is not in the paper but is an experiment I should try. Maybe let it pay attention to one or the other based on the past input? But we are still not letting it chose when to think.... hmm
- 
```
switch = F.sigmoid(nn.Linear(input_embeddings, 1)) # or whatever lstm and attn use
embeddings = input_embeddings * switch + thought_hidden * (1-switch)
```

# 2025-01-15 15:00:40

Ok I coded up a version that generated the input_embeds from the thought, then does a full forward pass. This is nice code, but poor gradient and GPU ram management.

With my 
model 7gb
stage 0: 17gb
stage 1: 23.3


For the better version I just need to do bot+i, and then I can do a forward pass with the cache. This is a bit more code, but should be faster and more memory efficient. I'll do this next.
Then I need to join all the outputs, which I can make a function for


Another idea is just to go step by step, with a batch, with recusion at each step, the transformer can switch to using it or not
