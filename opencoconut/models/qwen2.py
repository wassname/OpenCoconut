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
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Tuple, List, Union
from . import CoconutConfig
from ..dataset import split_sequences

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
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        num_thoughts: int = 1,
        tokens_per_thought: int = 2,
    ):
        """
        Generate continuous thought embeddings.
        """
        num_total_thoughts = num_thoughts * tokens_per_thought
        all_hidden_states = [inputs_embeds]
        all_thought_outputs = []
        current_mask = attention_mask
        current_labels = labels

        for t in range(num_total_thoughts):
            # Generate next thought embedding
            outputs = self.model.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

            # Extract and store next hidden state
            next_hidden = outputs.last_hidden_state[:, -1:, :]
            all_hidden_states.append(next_hidden)

            # Extend attention mask and labels
            if current_mask is not None:
                current_mask = append_tensor(current_mask, 1)
            if current_labels is not None:
                current_labels = append_tensor(current_labels, self.ignore_label)
            
            past_key_values = outputs.past_key_values


        if self.debug:

            all_hidden_states
            all_thought_outputs.append(
                self.hidden_states_to_token(inputs_embeds, lm_head=True)
            )

        return (
            torch.cat(all_hidden_states, dim=1),
            current_mask,
            current_labels,
            past_key_values
        )

    # def append_bot(self, input_ids, attention_mask):
    #     """append beginning of thought token."""

    #     input_ids = append_tensor(input_ids, self.coconut_config.bot_id)
    #     if attention_mask is not None:
    #         attention_mask = append_tensor(attention_mask, 1)
    #     input_ids = torch.concat(
    #         [
    #             input_ids,
    #             torch.tensor(
    #                 [[self.coconut_config.bot_id]] * input_ids.shape[0],
    #                 device=input_ids.device,
    #             ),
    #         ],
    #         dim=1,
    #     )
    #     if attention_mask is not None:
    #         attention_mask = torch.concat(
    #             [
    #                 attention_mask,
    #                 torch.ones(
    #                     attention_mask.shape[0], 1, device=attention_mask.device
    #                 ),
    #             ],
    #             dim=1,
    #         )
    #     return input_ids, attention_mask

    def append_bot_token(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        """Append BOT token if not present."""
        if input_ids.shape[1] > 1:
            # input_ids = torch.cat([
            #     input_ids,
            #     torch.full(
            #         (input_ids.shape[0], 1),
            #         self.config.bot_token_id,
            #         device=input_ids.device
            #     )
            # ], dim=1)
            # FIXME can't I usee append_tensor here?
            input_ids = append_tensor(input_ids, self.config.bot_token_id)
            if attention_mask is not None:
                attention_mask = append_tensor(attention_mask, 1)
        return input_ids, attention_mask
    
    def inference_forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        num_thoughts: Optional[int] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """Inference with continuous thought generation."""
        batch_size = input_ids.shape[0]
        num_thoughts = num_thoughts or (self.current_stage * self.config.continuous_thoughts)
        
        # Inject BOT token if needed
        input_ids, attention_mask = self.append_bot_token(input_ids, attention_mask)
        
        # Initial context
        context_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True
        )

        # Generate thoughts using shared implementation
        thought_hidden, thought_mask, _, past_kv = self.thoughts_forward(
            hidden_states=context_outputs.last_hidden_state[:, -1:, :],
            attention_mask=attention_mask,
            past_key_values=context_outputs.past_key_values,
            num_thoughts=num_thoughts,
            tokens_per_thought=self.config.continuous_thoughts,
            temperature=temperature
        )

        # Append EOT and generate final output
        eot_ids = append_tensor(
            torch.empty(batch_size, 0, dtype=torch.long, device=input_ids.device),
            self.config.eot_token_id,
            num=1
        )
        
        suffix_outputs = super().forward(
            input_ids=eot_ids,
            attention_mask=append_tensor(thought_mask, 1),
            past_key_values=past_kv,
            **kwargs
        )

        return self.combine_triple_outputs(
            context_outputs, thought_hidden, suffix_outputs, kwargs
        )

    
    def combine_triple_outputs(self, prefix_outputs, thought_hidden, suffix_outputs, loss=None, **kwargs):
        """Combine prefix, thought and suffix outputs."""

        if kwargs.get("output_hidden_states", False):
            hidden_states = prefix_outputs.hidden_states + [thought_hidden] + suffix_outputs.hidden_states
        else:
            hidden_states = None

        if kwargs.get("output_attentions", False):
            raise NotImplementedError("output_attentions not implemented")
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=suffix_outputs.logits,
            past_key_values=suffix_outputs.past_key_values,
            hidden_states=hidden_states,
        )
    
    # def infer_forward(
    #     self,
    #     input_ids: torch.LongTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     past_key_values: Optional[Union[DynamicCache, List[torch.FloatTensor]]] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     **loss_kwargs,
    # ):
    #     batch_size = input_ids.shape[0]

    #     if input_ids.shape[1] > 1:
    #         input_ids = append_tensor(input_ids, self.coconut_config.bot_id)
    #         if attention_mask is not None:
    #             attention_mask = append_tensor(attention_mask, 1)

    #     if past_key_values is None:
    #         past_key_values = DynamicCache()

    #     # NOTE: only generate thoughts in the prefilling phase
    #     if self.coconut_config.stages - 1 > 0 and input_ids.shape[1] > 1:
    #         num_thoughts = (
    #             (self.coconut_config.stages - 1) * self.coconut_config.continuous_thoughts
    #         )
    #         inputs_embeds = self.get_input_embeddings()(input_ids)

    #         all_thought_outputs = self.thoughts_forward(
    #             num_thoughts, inputs_embeds, attention_mask, past_key_values
    #         )

    #         language_ids = torch.tensor(
    #             [[self.coconut_config.eot_id]] * batch_size,
    #             device=inputs_embeds.device,
    #         )
    #         inputs_embeds = self.get_input_embeddings()(language_ids)

    #         # we fix the mask and labels lengths by inserting between <bot><eot>
    #         insert_indices = (input_ids == self.coconut_config.eot_id).nonzero(
    #             as_tuple=True
    #         )[1]

    #         new_attention_mask = []
    #         for b in range(batch_size):

    #             # append the thought tokens to the attention mask
    #             new_attention_mask.append(
    #                 torch.cat(
    #                     (
    #                         attention_mask[b],
    #                         torch.ones(
    #                             num_thoughts - 1,
    #                             dtype=attention_mask.dtype,
    #                             device=attention_mask.device,
    #                         ),
    #                     )
    #                 )
    #             )
    #         attention_mask = torch.stack(new_attention_mask, dim=0)
            

    #         # Forward pass with combined embeddings
    #         outputs = super().forward(
    #             input_ids=None,
    #             attention_mask=attention_mask,
    #             position_ids=None,
    #             past_key_values=past_key_values,
    #             inputs_embeds=inputs_embeds,
    #             labels=labels,
    #             use_cache=True,
    #             # output_attentions=output_attentions,
    #             output_hidden_states=True,
    #             return_dict=True,
    #             # num_logits_to_keep=num_logits_to_keep,
    #         )

    #         if self.debug:
    #             self._print_thought_and_final_tokens(
    #                 outputs.logits, all_thought_outputs
    #             )

    #     else:
    #         # Standard forward pass
    #         outputs = super().forward(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             past_key_values=past_key_values,
    #             inputs_embeds=None,
    #             labels=labels,
    #             return_dict=True,
    #             **loss_kwargs,
    #         )

    #     return outputs

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
        bot_pos = (input_ids == self.config.bot_token_id).nonzero()[:, 1][0]
        eot_pos = (input_ids == self.config.eot_token_id).nonzero()[:, 1][0]
        
        # Process prefix sequence (including BOT token)
        prefix_outputs = super().forward(
            input_ids=input_ids[:, :bot_pos+1],
            attention_mask=attention_mask[:, :bot_pos+1],
            labels=labels[:, :bot_pos+1] if labels is not None else None,
            use_cache=True,
            return_dict=True
        )

        # Generate thoughts
        thought_hidden, thought_mask, thought_labels, past_kv = self.thoughts_forward(
            hidden_states=prefix_outputs.last_hidden_state[:, -1:, :],
            attention_mask=attention_mask[:, :bot_pos+1],
            labels=labels[:, :bot_pos+1] if labels is not None else None,
            past_key_values=prefix_outputs.past_key_values,
            num_thoughts=self.current_stage,
            tokens_per_thought=self.config.continuous_thoughts
        )

        # Process suffix (after EOT)
        suffix_outputs = super().forward(
            input_ids=input_ids[:, eot_pos:],
            attention_mask=attention_mask[:, eot_pos:],
            past_key_values=past_kv, # Contains both prefix AND thought contexts
            labels=labels[:, eot_pos:] if labels is not None else None,
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

    @torch.no_grad()
    def hidden_states_to_token(self, logits: torch.Tensor, lm_head=False):
        if lm_head:
            logits = self.lm_head(logits)
        probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
        top_probs, top_indices = torch.topk(probs, 3)

        tokens = []

        for prob, token_id in zip(top_probs.squeeze(), top_indices.squeeze()):
            tokens.append(
                {
                    "token": self.tokenizer.decode(token_id.item()),
                    "prob": prob.item(),
                    "token_id": token_id.item(),
                }
            )

        return tokens

    def _print_thought_and_final_tokens(
        self, logits: torch.Tensor, all_thought_outputs: List[torch.Tensor]
    ):
        final_thoughts = []
        final_token = self.hidden_states_to_token(logits)[0]
        for i, sampled_tokens in enumerate(all_thought_outputs):
            tokens_formatted = []
            for j, token in enumerate(sampled_tokens):
                tokens_formatted.append(
                    f"t_{i},{j}: [{token['token'].strip()}] (p: {token['prob']:.3f})"
                )
            final_thoughts.append((" || ").join(tokens_formatted))
        print("\n".join(final_thoughts))
        print(
            f"t_final: [{final_token['token'].strip()}] (p: {final_token['prob']:.3f})"
        )
