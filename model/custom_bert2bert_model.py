from abc import ABC
import torch
from transformers.models.encoder_decoder.modeling_encoder_decoder import EncoderDecoderModel
from transformers.models.mbart import MBartForConditionalGeneration, MBartConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.mbart.modeling_mbart import (
    MBART_INPUTS_DOCSTRING,
    _CONFIG_FOR_DOC,
    MBART_GENERATION_EXAMPLE,
    shift_tokens_right
)
from torch.nn import CrossEntropyLoss
import numpy as np


class CustomBERT2BERTModel(EncoderDecoderModel, ABC):
    def __init__(self, config=None, encoder=None, decoder=None):
        super(CustomBERT2BERTModel, self).__init__(config=config, encoder=encoder, decoder=decoder)
        self.word_dropout_ratio = None
        self.word_replacement_ratio = None

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
                    batch[batch_idx][token_idx] = self.config.mask_token_id
        return batch

    def word_replacement(self, batch_ids_):
        if self.word_replacement_ratio == 0:
            return batch_ids_
        batch_ids = torch.clone(batch_ids_)
        embedding_size = len(self.encoder.embeddings.word_embeddings.weight)
        for batch_idx in range(len(batch_ids)):
            for token_idx in range(len(batch_ids[batch_idx])):
                if token_idx == self.config.pad_token_id:
                    break
                if np.random.randn(1)[0] < self.word_replacement_ratio:
                    batch_ids[batch_idx][token_idx] = np.random.randint(0, embedding_size)
        return batch_ids

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if labels is not None:
            input_ids = self.word_replacement(input_ids)
            input_ids = self.word_dropout(input_ids)


        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )

        encoder_hidden_states = encoder_outputs[0]

        # optionally project encoder_hidden_states
        if (
                self.encoder.config.hidden_size != self.decoder.config.hidden_size
                and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = self.word_replacement(labels)
            _decoder_input_ids = self.word_dropout(decoder_input_ids)
            decoder_input_ids = shift_tokens_right(
                _decoder_input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
