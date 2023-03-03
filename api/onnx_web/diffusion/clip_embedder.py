from typing import List
import torch
from torch import nn
from transformers import CLIPTokenizer, CLIPTextModel
from functools import partial

###
# From:
# - https://github.com/rinongal/textual_inversion/blob/main/ldm/modules/encoders/modules.py
# - https://github.com/rinongal/textual_inversion/blob/main/ldm/modules/embedding_manager.py
# - https://github.com/rinongal/textual_inversion/blob/main/merge_embeddings.py
###

def _expand_mask(mask, dtype, tgt_len=None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


def _build_causal_attention_mask(bsz, seq_len, dtype):
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
    mask.fill_(torch.tensor(torch.finfo(dtype).min))
    mask.triu_(1)  # zero out the lower diagonal
    mask = mask.unsqueeze(1)  # expand mask
    return mask


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(
        self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77
    ):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        # self.freeze()

        def embedding_forward(
            self,
            input_ids=None,
            position_ids=None,
            inputs_embeds=None,
            embedding_manager=None,
        ) -> torch.Tensor:
            seq_length = (
                input_ids.shape[-1]
                if input_ids is not None
                else inputs_embeds.shape[-2]
            )

            if position_ids is None:
                position_ids = self.position_ids[:, :seq_length]

            if inputs_embeds is None:
                inputs_embeds = self.token_embedding(input_ids)

            if embedding_manager is not None:
                inputs_embeds = embedding_manager(input_ids, inputs_embeds)

            position_embeddings = self.position_embedding(position_ids)
            embeddings = inputs_embeds + position_embeddings

            return embeddings

        self.transformer.text_model.embeddings.forward = embedding_forward.__get__(
            self.transformer.text_model.embeddings
        )

        def encoder_forward(
            self,
            inputs_embeds,
            attention_mask=None,
            causal_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        ):
            output_attentions = (
                output_attentions
                if output_attentions is not None
                else self.config.output_attentions
            )
            output_hidden_states = (
                output_hidden_states
                if output_hidden_states is not None
                else self.config.output_hidden_states
            )
            return_dict = (
                return_dict if return_dict is not None else self.config.use_return_dict
            )

            encoder_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None

            hidden_states = inputs_embeds
            for idx, encoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states,)

                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            return hidden_states

        self.transformer.text_model.encoder.forward = encoder_forward.__get__(
            self.transformer.text_model.encoder
        )

        def text_encoder_forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            embedding_manager=None,
        ):
            output_attentions = (
                output_attentions
                if output_attentions is not None
                else self.config.output_attentions
            )
            output_hidden_states = (
                output_hidden_states
                if output_hidden_states is not None
                else self.config.output_hidden_states
            )
            return_dict = (
                return_dict if return_dict is not None else self.config.use_return_dict
            )

            if input_ids is None:
                raise ValueError("You have to specify either input_ids")

            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

            hidden_states = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                embedding_manager=embedding_manager,
            )

            bsz, seq_len = input_shape
            # CLIP's text model uses causal mask, prepare it here.
            # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
            causal_attention_mask = _build_causal_attention_mask(
                bsz, seq_len, hidden_states.dtype
            ).to(hidden_states.device)

            # expand attention_mask
            if attention_mask is not None:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

            last_hidden_state = self.encoder(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            last_hidden_state = self.final_layer_norm(last_hidden_state)

            return last_hidden_state

        self.transformer.text_model.forward = text_encoder_forward.__get__(
            self.transformer.text_model
        )

        def transformer_forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            embedding_manager=None,
        ):
            return self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                embedding_manager=embedding_manager,
            )

        self.transformer.forward = transformer_forward.__get__(self.transformer)

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text, **kwargs):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        z = self.transformer(input_ids=tokens, **kwargs)

        return z

    def encode(self, text, **kwargs):
        return self(text, **kwargs)


DEFAULT_PLACEHOLDER_TOKEN = ["*"]

PROGRESSIVE_SCALE = 2000


def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(
        string,
        truncation=True,
        max_length=77,
        return_length=True,
        return_overflowing_tokens=False,
        padding="max_length",
        return_tensors="pt",
    )
    tokens = batch_encoding["input_ids"]
    # assert torch.count_nonzero(tokens - 49407) == 2, f"String '{string}' maps to more than a single token. Please use another string"

    return tokens[0, 1]


def get_bert_token_for_string(tokenizer, string):
    token = tokenizer(string)
    assert (
        torch.count_nonzero(token) == 3
    ), f"String '{string}' maps to more than a single token. Please use another string"

    token = token[0, 1]

    return token


def get_embedding_for_clip_token(embedder, token):
    return embedder(token.unsqueeze(0))[0, 0]


class EmbeddingManager(nn.Module):
    def __init__(
        self,
        embedder,
        placeholder_strings=None,
        initializer_words=None,
        per_image_tokens=False,
        num_vectors_per_token=1,
        progressive_words=False,
        **kwargs,
    ):
        super().__init__()

        self.string_to_token_dict = {}

        self.string_to_param_dict = nn.ParameterDict()

        self.initial_embeddings = nn.ParameterDict()  # These should not be optimized

        self.progressive_words = progressive_words
        self.progressive_counter = 0

        self.max_vectors_per_token = num_vectors_per_token

        if hasattr(embedder, "tokenizer"):  # using Stable Diffusion's CLIP encoder
            self.is_clip = True
            get_token_for_string = partial(
                get_clip_token_for_string, embedder.tokenizer
            )
            get_embedding_for_tkn = partial(
                get_embedding_for_clip_token, embedder.transformer.text_model.embeddings
            )
            token_dim = 768
        else:  # using LDM's BERT encoder
            self.is_clip = False
            get_token_for_string = partial(get_bert_token_for_string, embedder.tknz_fn)
            get_embedding_for_tkn = embedder.transformer.token_emb
            token_dim = 1280

        # if per_image_tokens:
        #     placeholder_strings.extend(per_img_token_list)

        for idx, placeholder_string in enumerate(placeholder_strings):
            token = get_token_for_string(placeholder_string)

            if initializer_words and idx < len(initializer_words):
                init_word_token = get_token_for_string(initializer_words[idx])

                with torch.no_grad():
                    init_word_embedding = get_embedding_for_tkn(init_word_token.cpu())

                token_params = torch.nn.Parameter(
                    init_word_embedding.unsqueeze(0).repeat(num_vectors_per_token, 1),
                    requires_grad=True,
                )
                self.initial_embeddings[placeholder_string] = torch.nn.Parameter(
                    init_word_embedding.unsqueeze(0).repeat(num_vectors_per_token, 1),
                    requires_grad=False,
                )
            else:
                token_params = torch.nn.Parameter(
                    torch.rand(
                        size=(num_vectors_per_token, token_dim), requires_grad=True
                    )
                )

            self.string_to_token_dict[placeholder_string] = token
            self.string_to_param_dict[placeholder_string] = token_params

    def forward(
        self,
        tokenized_text,
        embedded_text,
    ):
        b, n, device = *tokenized_text.shape, tokenized_text.device

        for placeholder_string, placeholder_token in self.string_to_token_dict.items():
            placeholder_embedding = self.string_to_param_dict[placeholder_string].to(
                device
            )

            if (
                self.max_vectors_per_token == 1
            ):  # If there's only one vector per token, we can do a simple replacement
                placeholder_idx = torch.where(
                    tokenized_text == placeholder_token.to(device)
                )
                embedded_text[placeholder_idx] = placeholder_embedding
            else:  # otherwise, need to insert and keep track of changing indices
                if self.progressive_words:
                    self.progressive_counter += 1
                    max_step_tokens = 1 + self.progressive_counter // PROGRESSIVE_SCALE
                else:
                    max_step_tokens = self.max_vectors_per_token

                num_vectors_for_token = min(
                    placeholder_embedding.shape[0], max_step_tokens
                )

                placeholder_rows, placeholder_cols = torch.where(
                    tokenized_text == placeholder_token.to(device)
                )

                if placeholder_rows.nelement() == 0:
                    continue

                sorted_cols, sort_idx = torch.sort(placeholder_cols, descending=True)
                sorted_rows = placeholder_rows[sort_idx]

                for idx in range(len(sorted_rows)):
                    row = sorted_rows[idx]
                    col = sorted_cols[idx]

                    new_token_row = torch.cat(
                        [
                            tokenized_text[row][:col],
                            placeholder_token.repeat(num_vectors_for_token).to(device),
                            tokenized_text[row][col + 1 :],
                        ],
                        axis=0,
                    )[:n]
                    new_embed_row = torch.cat(
                        [
                            embedded_text[row][:col],
                            placeholder_embedding[:num_vectors_for_token],
                            embedded_text[row][col + 1 :],
                        ],
                        axis=0,
                    )[:n]

                    embedded_text[row] = new_embed_row
                    tokenized_text[row] = new_token_row

        return embedded_text

    def save(self, ckpt_path):
        torch.save(
            {
                "string_to_token": self.string_to_token_dict,
                "string_to_param": self.string_to_param_dict,
            },
            ckpt_path,
        )

    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")

        del self.string_to_param_dict
        del self.string_to_token_dict

        self.string_to_token_dict = ckpt["string_to_token"]
        self.string_to_param_dict = ckpt["string_to_param"]

    def get_embedding_norms_squared(self):
        all_params = torch.cat(
            list(self.string_to_param_dict.values()), axis=0
        )  # num_placeholders x embedding_dim
        param_norm_squared = (all_params * all_params).sum(axis=-1)  # num_placeholders

        return param_norm_squared

    def embedding_parameters(self):
        return self.string_to_param_dict.parameters()

    def embedding_to_coarse_loss(self):
        loss = 0.0
        num_embeddings = len(self.initial_embeddings)

        for key in self.initial_embeddings:
            optimized = self.string_to_param_dict[key]
            coarse = self.initial_embeddings[key].clone().to(optimized.device)

            loss = loss + (optimized - coarse) @ (optimized - coarse).T / num_embeddings

        return loss


def get_placeholder_loop(placeholder_string, embedder):
    new_placeholder = None

    while True:
        if new_placeholder is None:
            new_placeholder = input(
                f"Placeholder string {placeholder_string} was already used. Please enter a replacement string: "
            )
        else:
            new_placeholder = input(
                f"Placeholder string '{new_placeholder}' maps to more than a single token. Please enter another string: "
            )

        token = get_clip_token_for_string(embedder.tokenizer, new_placeholder)

        if token is not None:
            return new_placeholder, token


def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(
        string,
        truncation=True,
        max_length=77,
        return_length=True,
        return_overflowing_tokens=False,
        padding="max_length",
        return_tensors="pt",
    )
    tokens = batch_encoding["input_ids"]

    if torch.count_nonzero(tokens - 49407) == 2:
        return tokens[0, 1]

    return None


def load_tokenizer(manager_ckpts: List[str]):
    embedder = FrozenCLIPEmbedder().cuda()
    FrozenEmbeddingManager = partial(EmbeddingManager, embedder, ["*"])

    string_to_token_dict = {}
    string_to_param_dict = torch.nn.ParameterDict()

    placeholder_to_src = {}

    for manager_ckpt in manager_ckpts:
        print(f"Parsing {manager_ckpt}...")

        manager = FrozenEmbeddingManager()
        manager.load(manager_ckpt)

        for placeholder_string in manager.string_to_token_dict:
            if not placeholder_string in string_to_token_dict:
                string_to_token_dict[placeholder_string] = manager.string_to_token_dict[
                    placeholder_string
                ]
                string_to_param_dict[placeholder_string] = manager.string_to_param_dict[
                    placeholder_string
                ]

                placeholder_to_src[placeholder_string] = manager_ckpt
            else:
                new_placeholder, new_token = get_placeholder_loop(
                    placeholder_string, embedder
                )
                string_to_token_dict[new_placeholder] = new_token
                string_to_param_dict[new_placeholder] = manager.string_to_param_dict[
                    placeholder_string
                ]

                placeholder_to_src[new_placeholder] = manager_ckpt

    print("Saving combined manager...")
    merged_manager = FrozenEmbeddingManager()
    merged_manager.string_to_param_dict = string_to_param_dict
    merged_manager.string_to_token_dict = string_to_token_dict

    print("Managers merged. Final list of placeholders: ")
    print(placeholder_to_src)

    return merged_manager

