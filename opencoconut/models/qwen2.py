import os
import torch
import torch.nn.functional as F
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
from einops import rearrange

logger = logging.getLogger(__name__)

def append_tensor(t: torch.Tensor, value: Union[int, float], num: int = 1) -> torch.Tensor:
    """Append value n times to the sequence dimension of tensor."""
    return F.pad(t, (0, num), value=value)


class CoconutQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer: PreTrainedTokenizer = None
        self.coconut_config: CoconutConfig = CoconutConfig.from_dict(
            config.coconut_config
        )
        self.current_stage = 0
        self.debug = os.environ.get("DEBUG") == "1"

    def thoughts_forward(
        self,
        inputs_embeds: torch.FloatTensor,
        # attention_mask: Optional[torch.Tensor] = None,
        # labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        num_thoughts: int = 1,
        tokens_per_thought: int = 2,
    ):
        """
        Generate continuous thought embeddings.
        """
        num_total_thoughts = num_thoughts * tokens_per_thought
        all_hidden_states = []
        all_thought_outputs = []
        # current_mask = attention_mask
        # current_labels = labels

        # past_seen_tokens = past_key_values.get_seq_length()


        def copy_cache(cache: DynamicCache):
            new = DynamicCache()
            new._seen_tokens = cache._seen_tokens
            new.key_cache = [
                k.clone()
                for k in cache.key_cache
            ]
            new.value_cache = [
                v.clone()
                for v in cache.value_cache
            ]
            return new


        for t in range(num_total_thoughts):

            current_kv = copy_cache(past_key_values)
            # cache_position = torch.arange(
            #     past_seen_tokens + t, past_seen_tokens + t + 1, device=inputs_embeds.device
            # )
            # cache_position = torch.arange(
            #     0, 1, device=inputs_embeds.device
            # )
            print(f'kv.{t}-a', past_key_values[0][0].shape, past_key_values.seen_tokens)
            # Generate next thought embedding

            # TODO try with super().forward
            outputs = self.model.forward(
                inputs_embeds=inputs_embeds,
                # attention_mask=current_mask,
                past_key_values=current_kv,
                # cache_position=cache_position,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )

            # TODO add just last cache to past_key_values
            past_key_values.key_cache.append(outputs.past_key_values.key_cache[-1])
            past_key_values.value_cache.append(outputs.past_key_values.value_cache[-1])
            past_key_values.seen_tokens += 1

            # Extract and store next hidden state
            inputs_embeds = outputs.last_hidden_state[:, -1:, :]
            hs = torch.stack(outputs.hidden_states)[:, :, -1, :] # [layer, batch, seq, hidden]
            all_hidden_states.append(hs)

            # # Extend attention mask and labels
            # if current_mask is not None:
            #     current_mask = append_tensor(current_mask, 1)
            # if current_labels is not None:
            #     current_labels = append_tensor(current_labels, self.ignore_label)
            print(f'kv.{t}-b', past_key_values[0][0].shape, past_key_values.seen_tokens)
            print(f'kv.{t}-c', outputs.past_key_values[0][0].shape)
            
            past_key_values = outputs.past_key_values


        # if self.debug:

        #     all_hidden_states
        #     all_thought_outputs.append(
        #         self.hidden_states_to_token(inputs_embeds, lm_head=True)
        #     )

        # hidden states are meant to be tuple([layer, batch, seq, hidden]), but this is [seq, layer, batch, hidden]
        # so we need to transpose using einops
        if all_hidden_states:
            all_hidden_states = rearrange(all_hidden_states, 's l b h -> l b s h')


        return (
            tuple(all_hidden_states),
            # current_mask,
            # current_labels,
            past_key_values
        )


    def append_bot_token(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        """Append BOT token if not present."""
        if input_ids.shape[1] > 1:
            input_ids = append_tensor(input_ids, self.coconut_config.bot_id)
            if attention_mask is not None:
                attention_mask = append_tensor(attention_mask, 1)
        return input_ids, attention_mask
    
    def infer_forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
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
        thought_hidden, past_kv = self.thoughts_forward(
            inputs_embeds=context_outputs.last_hidden_state[:, -1:, :],
            # attention_mask=attention_mask,
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

    
    def combine_triple_outputs(self, prefix_outputs: BaseModelOutputWithPast, thought_hidden: Tuple[torch.Tensor], suffix_outputs: CausalLMOutputWithPast, loss=None, **kwargs):
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
        attention_mask: Optional[torch.Tensor] = None,
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
        bot_pos = (input_ids == self.coconut_config.bot_id).nonzero()[:, 1][0]
        eot_pos = (input_ids == self.coconut_config.eot_id).nonzero()[:, 1][0]
        
        # Process prefix sequence (including BOT token)
        prefix_outputs = self.model.forward(
            input_ids=input_ids[:, :bot_pos+1],
            attention_mask=attention_mask[:, :bot_pos+1],
            labels=labels[:, :bot_pos+1] if labels is not None else None,
            use_cache=True,
            return_dict=True,
            return_hidden_states=True,
        )
        print('kv0', prefix_outputs.past_key_values[0][0].shape)

        # Generate thoughts
        # FIXME this is doubling kv cache size...
        thought_hidden, past_kv = self.thoughts_forward(
            inputs_embeds=prefix_outputs.last_hidden_state,
            # attention_mask=attention_mask[:, :bot_pos+1],
            # labels=labels[:, :bot_pos+1] if labels is not None else None,
            past_key_values=prefix_outputs.past_key_values,
            num_thoughts=self.current_stage,
            tokens_per_thought=self.coconut_config.continuous_thoughts
        )
        print('kv1', prefix_outputs.past_key_values[0][0].shape)
        print('kv2', past_kv[0][0].shape)

        # Process suffix (after EOT)
        suffix_outputs = self.model.forward(
            input_ids=input_ids[:, eot_pos:],
            attention_mask=attention_mask[:, eot_pos:],
            past_key_values=past_kv, # Contains both prefix AND thought contexts
            labels=labels[:, eot_pos:] if labels is not None else None,
            return_hidden_states=True,
            **kwargs
        )


        # Update total loss
        if labels is not None:
            # weight by tokens in each part
            total_loss = (
                prefix_outputs.loss * bot_pos + 
                suffix_outputs.loss * (input_ids.shape[1] - eot_pos)
            ) / (bot_pos + input_ids.shape[1] - eot_pos)
        else:
            total_loss = None

        return self.combine_triple_outputs(
            prefix_outputs, thought_hidden, suffix_outputs, loss=total_loss, **kwargs
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
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
