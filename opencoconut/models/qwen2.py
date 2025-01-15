import os
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from jaxtyping import Float, Int, Bool
from einops import rearrange, reduce, repeat, parse_shape, pack, unpack

import logging
from typing import Optional, List, Union
from transformers import (
    Qwen2ForCausalLM,
    DynamicCache,
    PreTrainedTokenizer,
)
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from typing import Tuple, List, Union
from . import CoconutConfig
from ..dataset import split_sequences

logger = logging.getLogger(__name__)

def append_tensor(t: Tensor, value: Union[int, float], num: int = 1) -> Tensor:
    """Append value n times to the sequence dimension of tensor."""
    return F.pad(t, (0, num), value=value)

# when the Qwen model has output_hidden_states=True, the hidden_states is a tuple of hidden states from each layer
HS = Tuple[Float[Tensor, 'b s h']]

class CoconutQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer: PreTrainedTokenizer = None

        # FIXME probobly better to subclass Qwen2Config
        self.coconut_config: CoconutConfig = CoconutConfig.from_dict(
            config.coconut_config
        )
        self.current_stage = 0
        self.debug = os.environ.get("DEBUG") == "1"


    def hs2embed(self, hidden_states: HS) -> Float[Tensor, 'b 1 h']:
        """Convert hidden states to input embeddings."""
        # TODO try experiments here, like only using supressed neurons, residual, etc https://github.com/wassname/repr-preference-optimization/tree/main/reprpo/interventions/transforms

        """
        Novel experiment: Here we define a transform to isolate supressed activations, where we hypothesis that style/concepts/scratchpads and other internal only representations must be stored.

        See the following references for more information:
        - https://arxiv.org/html/2406.19384v1
            - > Previous work suggests that networks contain ensembles of “prediction" neurons, which act as probability promoters [66, 24, 32] and work in tandem with suppression neurons (Section 5.4). 

        - https://arxiv.org/pdf/2401.12181
            > We find a striking pattern which is remarkably consistent across the different seeds: after about the halfway point in the model, prediction neurons become increasingly prevalent until the very end of the network where there is a sudden shift towards a much larger number of suppression neurons.
        """
        # supressed_act = rearrange(list(hidden_states), 'l b 1 h -> l b h').diff(dim=0).clamp(min=None, max=0)
        # hs = supressed_act[-1][:, None] # last layer, add dummy sequence dim

        # just get last layer, last token
        hs = hidden_states[-1][:, -1:, :]

        return hs

    def thoughts_forward(
        self,
        inputs_embeds: torch.FloatTensor,
        past_key_values: Optional[DynamicCache] = None,
        num_thoughts: int = 1,
        tokens_per_thought: int = 2,
    ):
        """
        Generate continuous thought embeddings.

        If given mask and labels we insert into these as we produce additional input_embeddings
        """
        num_total_thoughts = num_thoughts * tokens_per_thought
        all_hidden_states = []
        new_input_embeds = []

        for t in range(num_total_thoughts):
            # Generate next thought embedding

            outputs = self.model.forward(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )

            past_key_values = outputs.past_key_values
            # Extract and store next hidden state
            inputs_embeds = self.hs2embed(outputs.hidden_states)
            # TODO do I need last masked... nah
            last_hs = rearrange(list(outputs.hidden_states), 'l b 1 h -> l b h')
            all_hidden_states.append(last_hs)

            new_input_embeds.append(inputs_embeds)        


        # hidden states are meant to be tuple([layer, batch, seq, hidden]), but this is [seq, layer, batch, hidden]
        # so we need to transpose using einops
        if all_hidden_states:
            all_hidden_states = rearrange(all_hidden_states, 's l b h -> l b s h')

        if new_input_embeds:
            new_input_embeds = torch.cat(new_input_embeds, dim=1)


        return (
            tuple(all_hidden_states),
            new_input_embeds,
            past_key_values
        )


    def append_bot_token(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[torch.LongTensor, Tensor]:
        """Append BOT token if not present."""
        if input_ids.shape[1] > 1:
            input_ids = append_tensor(input_ids, self.coconut_config.bot_id)
            if attention_mask is not None:
                attention_mask = append_tensor(attention_mask, 1)
        return input_ids, attention_mask
    
    def infer_forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Tensor,
        num_thoughts: Optional[int] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """Inference with continuous thought generation."""
        batch_size = input_ids.shape[0]
        num_thoughts = num_thoughts or (self.current_stage * self.coconut_config.continuous_thoughts)
        
        # Inject BOT token if needed
        input_ids, attention_mask = self.append_bot_token(input_ids, attention_mask)
        
        # Initial context
        context_outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
            **kwargs
        )

        # Generate thoughts using shared implementation
        thought_hidden, _, past_kv = self.thoughts_forward(
            inputs_embeds=context_outputs.last_hidden_state[:, -1:, :],
            # attention_mask=attention_mask,
            # labels=labels,
            past_key_values=context_outputs.past_key_values,
            num_thoughts=num_thoughts,
            tokens_per_thought=self.coconut_config.continuous_thoughts,
            # temperature=temperature
        )

        # Append EOT and generate final output
        eot_ids = append_tensor(
            input_ids[:, -1:-1],
            self.coconut_config.eot_id
        )
        
        del kwargs['labels']
        # here we use the kv cache from the last thought
        suffix_outputs = super().forward(
            input_ids=eot_ids,
            # attention_mask=append_tensor(thought_mask, 1),
            past_key_values=past_kv,
            **kwargs
        )

        return self.combine_triple_outputs(
            context_outputs, 
            thought_hidden, 
            suffix_outputs, 
            **kwargs
        )

    
    def combine_triple_outputs(self, prefix_outputs: BaseModelOutputWithPast, thought_hidden: Tuple[Tensor], suffix_outputs: CausalLMOutputWithPast, loss=None, **kwargs):
        """Combine prefix, thought and suffix outputs."""

        if kwargs.get("output_hidden_states", False):
            # hs shape [layer, batch, seq, hidden], we want to append by sequence
            # hidden_states = prefix_outputs.hidden_states + tuple(thought_hidden) + suffix_outputs.hidden_states

            thoughts = [prefix_outputs.hidden_states]
            if thought_hidden:
                thoughts.append(thought_hidden)
            thoughts.append(suffix_outputs.hidden_states)

            # print('3a', torch.stack(prefix_outputs.hidden_states).shape)
            # if thought_hidden:
            #     print(1, torch.stack(thought_hidden).shape)
            # print('3b', torch.stack(suffix_outputs.hidden_states).shape)

            hidden_states = []
            for i in range(len(prefix_outputs.hidden_states)):
                hidden_states.append(
                    torch.cat([x[i] for x in thoughts], dim=1)
                )
        else:
            hidden_states = None

        if kwargs.get("output_attentions", False):
            raise NotImplementedError("output_attentions not implemented")
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=getattr(suffix_outputs, 'logits', None),
            past_key_values=suffix_outputs.past_key_values,
            hidden_states=hidden_states,
        )
    

    def train_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        
        if self.current_stage == 0:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
        
        
        
        # Find BOT/EOT positions, for simplicity we assume there is only one in train
        eot_pos = (input_ids == self.coconut_config.eot_id).nonzero(as_tuple=True)
        mask_prefix = torch.arange(input_ids.shape[1], device=input_ids.device).expand(input_ids.shape[0], -1) <= eot_pos[1].unsqueeze(1)
        
        # Process prefix sequence (including BOT token)
        prefix_outputs = self.model.forward(
            input_ids=input_ids * mask_prefix,
            attention_mask=attention_mask * mask_prefix if attention_mask is not None else mask_prefix,
            use_cache=True,
            return_dict=True,
            output_hidden_states=True,
        )

        # Generate thoughts
        _, thought_embeds, _ = self.thoughts_forward(
            inputs_embeds=prefix_outputs.hidden_states[-1][:, -1:, :],
            past_key_values=prefix_outputs.past_key_values,
            num_thoughts=self.current_stage,
            tokens_per_thought=self.coconut_config.continuous_thoughts
        )

        # join prefix, thoughts, and suffix. This includes embeddings, attention, and labels
        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds2 = []
        attention_mask2 = []
        labels2 = []
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)
        for b in range(input_ids.shape[0]):
            pos = eot_pos[1][b]

            inputs_embeds2.append(
                torch.cat([
                    inputs_embeds[b, :pos+1], 
                    thought_embeds[b], 
                    inputs_embeds[b, pos+1:]
                ], dim=0)
            )
            n = thought_embeds[b].shape[0]
            attention_mask2.append(
                torch.cat([
                    attention_mask[b, :pos+1], 
                    attention_mask.new_ones(n), 
                    attention_mask[b, pos+1:]
                ], dim=0)
            )
            labels2.append(
                torch.cat([
                    labels[b, :pos+1], 
                    labels.new_full((n,), -100), 
                    labels[b, pos+1:]
                ], dim=0)
            )

        inputs_embeds2 = torch.stack(inputs_embeds2)
        attention_mask2 = torch.stack(attention_mask2)
        labels2 = torch.stack(labels2)
        # TODO check ordering bot, thoughts, eot

        # b= 0
        # s_input = self.tokenizer.decode(input_ids[b])
        # pos = eot_pos[1][b]
        # ids_from_embed = self.lm_head(inputs_embeds2).softmax(-1).argmax(-1)[b]
        # s_w_thought_pre = self.tokenizer.decode(ids_from_embed[:pos])
        # s_w_thought_pos = self.tokenizer.decode(ids_from_embed[pos:])
        # s_w_thought = self.tokenizer.decode(ids_from_embed, skip_special_tokens=False)
        # print(s_input)
        # print(s_w_thought_pre)
        # print(s_w_thought_pos)
        # print(s_w_thought)

        # # DEBUG print nonsensical thoughts
        # ids_from_embed = self.lm_head(thought_embeds).softmax(-1).argmax(-1)
        # print([self.tokenizer.decode(s) for s in ids_from_embed])

        # This is ineffecient, we do a whole new forward pass. And it's because the transformer library doesn't handle kv cache while doing multiple inputs for a forward pass
        # it's get even more complicated when we have to handle multiple batches, with multiple thoughts, this is the simplest way to code it

        # FIXME this is a simple way to write it... but the gradient seems complex. It might be better to it one token at a time and use the cache
        return super().forward(
            inputs_embeds=inputs_embeds2,
            attention_mask=attention_mask2,
            labels=labels2,
            **kwargs
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if self.training:
            forward_fn = self.train_forward
        else:
            forward_fn = self.infer_forward

        outputs = forward_fn(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        return outputs
