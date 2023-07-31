from abc import ABC
from typing import Optional, Tuple, Union
import torch
from transformers.modeling_outputs import Seq2SeqModelOutput, BaseModelOutput
from transformers.models.marian.modeling_marian import MarianModel
from transformers.models.marian.modeling_marian import MarianMTModel
from transformers.models.marian.configuration_marian import MarianConfig

from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.marian.modeling_marian import (
    MARIAN_INPUTS_DOCSTRING,
    _CONFIG_FOR_DOC,
    MARIAN_GENERATION_EXAMPLE,
    shift_tokens_right
)
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np

from transformers import MarianMTModel


class CustomMarianModel(MarianMTModel, ABC):
    def __init__(self, config: MarianConfig):
        super(CustomMarianModel, self).__init__(config)
        self.word_dropout_ratio = None
        self.word_replacement_ratio = None
        self.pad_id = config.pad_token_id
        self.bos_id = config.bos_token_id
        self.eos_id = config.eos_token_id

    def set_augment_config(self, word_dropout_ratio: float = 0, word_replacement_ratio: float = 0):
        self.word_dropout_ratio = word_dropout_ratio
        self.word_replacement_ratio = word_replacement_ratio

    def word_dropout(self, batch_):
        if self.word_dropout_ratio == 0:
            return batch_
        batch = torch.clone(batch_)
        for batch_idx in range(len(batch)):
            for token_idx in range(len(batch[batch_idx])):
                if np.random.randn(1)[0] < self.word_dropout_ratio:
                    batch[batch_idx][token_idx] *= 0
        return batch

    def word_replacement(self, batch_ids_):
        if self.word_replacement_ratio == 0:
            return batch_ids_
        batch_ids = torch.clone(batch_ids_)
        embedding_size = len(self.model.shared.weight)
        for batch_idx in range(len(batch_ids)):
            for token_idx in range(len(batch_ids[batch_idx])):
                if token_idx == self.pad_id:
                    break
                if np.random.randn(1)[0] < self.word_replacement_ratio:
                    batch_ids[batch_idx][token_idx] = np.random.randint(0, embedding_size)
        return batch_ids

    @add_start_docstrings_to_model_forward(MARIAN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(MARIAN_GENERATION_EXAMPLE)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            input_ids = self.word_replacement(input_ids)
            decoder_input_ids = self.word_replacement(labels)
            decoder_input_ids = shift_tokens_right(decoder_input_ids, pad_token_id=self.config.pad_token_id,
                                                   decoder_start_token_id=self.bos_id)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
