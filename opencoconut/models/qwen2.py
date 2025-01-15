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

        self.switch = nn.Sequential(
            nn.Linear(config.hidden_size, 1, bias=True),
            nn.Sigmoid(),
        )
        # start this as negative so we don't use much of the previous hidden state
        self.switch[0].bias.data.fill_(-10)



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
        supressed_act = -rearrange(list(hidden_states), 'l b 1 h -> l b h').diff(dim=0).clamp(min=None, max=0)
        hs = supressed_act[-1][:, None] # last layer, add dummy sequence dim

        # # we could also try just, last hs
        # hs = hidden_states[-1]

        return hs


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        """
        if 'cache_position' in kwargs:
            # raise NotImplementedError('cache_position')
            del kwargs['cache_position']
            del kwargs['past_key_values']


        if ('inputs_embeds' in kwargs) and (kwargs['inputs_embeds'] is not None):
            inputs_embeds = kwargs.pop('inputs_embeds')
        else:
            if 'inputs_embeds' in kwargs:
                del kwargs['inputs_embeds']
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if 'past_key_values' in kwargs:            
            kv_cache = kwargs.pop('past_key_values')
            if kv_cache.get_seq_length()>0:
                raise NotImplementedError('cant use gen right now as we cant use its prefil')
        else:
            kv_cache = DynamicCache()

        kwargs.pop('use_cache', None)
        kwargs.pop('output_hidden_states', None)

        
        # step forward token by token, adding the previous hidden state to the input, if the model chooses
        step_outputs =[]
        for t in range(input_ids.shape[1]):

            # or just self.model.forward if I don't need loss
            # print('t', t, kv_cache.get_seq_length(), kwargs)
            o = self.model.forward(
                inputs_embeds=inputs_embeds[:, t:t+1],

                # note: attention_mask always has to have the length: len(past_key_values) + len(input_ids) - https://github.com/huggingface/transformers/issues/16811#issuecomment-1101603158
                attention_mask=attention_mask[:, :t+1] if (t>0) else None,
                # labels=labels,
                past_key_values=kv_cache,
                output_hidden_states=True,
                use_cache=True,
                **kwargs,
            )

            kv_cache = o.past_key_values

            # Here's the unique part, we add on information from the previous hidden state, onto the next input_embed, if the model chooses
            prev_input_embed = self.hs2embed(o.hidden_states).squeeze(1)
            if t%10==0:
                prev_input_embed = prev_input_embed.detach() # detach  every N steps to prevent gradient explosion
            if t < inputs_embeds.shape[1] - 1:
                switch_output = self.switch(inputs_embeds[:, t])
                new_embed = (switch_output * prev_input_embed) + inputs_embeds[:, t+1]
                inputs_embeds = torch.cat([inputs_embeds[:, :t+1], new_embed.unsqueeze(1), inputs_embeds[:, t+2:]], dim=1)
            else:
                pass
                # raise NotImplementedError('gen/append')
            step_outputs.append(o)
            
            
            
        # combine outputs

        # rearrange hidden states so they are the normal tuple[tensor] in [layer, batch token hidden_size]
        hs = [rearrange(list(o.hidden_states), 'l b 1 h -> l b h') for o in step_outputs]
        hs = tuple(rearrange(hs, 't l b h -> l b t h'))

        last_layer_hs = hs[-1]
        logits = self.lm_head(last_layer_hs)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        
        o = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=kv_cache,
            hidden_states=hs,
        )
        return o
