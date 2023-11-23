from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Config, Wav2Vec2ConformerForCTC
from transformers.modeling_outputs import CausalLMOutput
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from transformers import (
    Wav2Vec2Processor,
)

class Wav2Vec2ForCTCV2(Wav2Vec2ForCTC):
    def __init__(self, config):
        super().__init__(config)

    def resize_lm_head(self, new_num_tokens):
        old_lm_head = self.lm_head
        # Build new lm head
        old_num_tokens, old_lm_head_dim = old_lm_head.weight.size()
        new_lm_head_shape = (old_lm_head_dim, new_num_tokens)
        has_new_lm_head_bias = old_lm_head.bias is not None
        new_lm_head = nn.Linear(*new_lm_head_shape, bias=has_new_lm_head_bias)
        new_lm_head = new_lm_head.to(
            old_lm_head.weight.device, dtype=old_lm_head.weight.dtype
        )
        self._init_weights(new_lm_head)
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)

        # initialize new lm head (in particular added tokens)
        self._init_weights(new_lm_head)

        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[
            :num_tokens_to_copy, :
        ]
        new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[
            :num_tokens_to_copy
        ]
        self.lm_head = new_lm_head


class Wav2Vec2ForCTCV3(Wav2Vec2ForCTC):
    def __init__(self, config):
        super().__init__(config)
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        self.lm_head_inter = nn.Linear(output_hidden_size, config.vocab_size)
        # self.dropout_inter = nn.Dropout(config.final_dropout)

    def resize_lm_head(self, new_num_tokens):
        old_lm_head = self.lm_head
        # Build new lm head
        old_num_tokens, old_lm_head_dim = old_lm_head.weight.size()
        new_lm_head_shape = (old_lm_head_dim, new_num_tokens)
        has_new_lm_head_bias = old_lm_head.bias is not None
        new_lm_head = nn.Linear(*new_lm_head_shape, bias=has_new_lm_head_bias)
        new_lm_head = new_lm_head.to(
            old_lm_head.weight.device, dtype=old_lm_head.weight.dtype
        )
        self._init_weights(new_lm_head)
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)

        # initialize new lm head (in particular added tokens)
        self._init_weights(new_lm_head)

        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[
            :num_tokens_to_copy, :
        ]
        new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[
            :num_tokens_to_copy
        ]
        self.lm_head = new_lm_head
        self.lm_head_inter = new_lm_head

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        #print(type(outputs), len(outputs[0]), len(outputs[1]), len(outputs[2]))
        hidden_states_inter = outputs[2][12]
        logits_inter = self.lm_head_inter(hidden_states_inter)

        loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            log_probs_inter = nn.functional.log_softmax(logits_inter, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                loss_inter = nn.functional.ctc_loss(
                    log_probs_inter,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                loss = 0.7*loss + 0.3*loss_inter

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
