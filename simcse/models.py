import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
from collections import namedtuple


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp





class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "mask", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == 'mask':
            ids = attention_mask.sum(1) - 3
            return last_hidden[range(last_hidden.shape[0]), ids]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type in ["cls", "mask"]:
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.endp = nn.Parameter(torch.rand(cls.model_args.prompt_len, config.hidden_size))

    cls.init_weights()


def cl_forward(cls,
    encoder,
    input_ids=None,
    prompt_mask=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    num_sent = input_ids.size(1)


    mlm_outputs = None

    inputs_embeds = encoder.embeddings.word_embeddings(input_ids)
    inputs_embeds_s0 = inputs_embeds[:, 0]
    inputs_embeds_s1 = inputs_embeds[:, 1:]
    attention_mask_s0 = attention_mask[:, 0]
    prompt_mask_s0 = prompt_mask[:, 0]

    ids = attention_mask_s0.sum(1) - 3

    l = cls.endp.shape[0]

    for i in range(l):
        inputs_embeds_s0[range(batch_size), ids - (l - i)] = cls.endp[i]

    inputs_embeds = torch.cat([inputs_embeds_s0.unsqueeze(1), 
                               inputs_embeds_s1], dim=1)

    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    inputs_embeds = inputs_embeds.view((batch_size * num_sent, -1, inputs_embeds.size(-1)))
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    prompt_mask = prompt_mask.view((-1, prompt_mask.size(-1))) # (bs * num_sent len)

    if token_type_ids is not None:
        token_type_ids_s0 = token_type_ids[:, 0] # (bs * num_sent, len)
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)
    else:
        token_type_ids_s0 = None

    # Get raw embeddings
    outputs = encoder(
        None,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    p_outputs = encoder(
        None,
        attention_mask=prompt_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids[:,0]
        mlm_input_embeds = encoder.embeddings.word_embeddings(mlm_input_ids)

        # mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            None,
            attention_mask=1-prompt_mask_s0,
            token_type_ids=token_type_ids_s0,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=mlm_input_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    p_pooler_output = cls.pooler(attention_mask, p_outputs)
    p_pooler_output = p_pooler_output.view((batch_size, num_sent, p_pooler_output.size(-1))) # (bs, num_sent, hidden)

    pooler_output = pooler_output - p_pooler_output

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type in ["cls", "mask"]:
        pooler_output = cls.mlp(pooler_output)


    # Separate representation
    z1, z2, z3, n1 = pooler_output[:, 0], pooler_output[:, 1], pooler_output[:,2], pooler_output[:,3]


    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
        n1_list = [torch.zeros_like(n1) for _ in range(dist.get_world_size())]

        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())
        dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
        dist.all_gather(tensor_list=n1_list, tensor=n1.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        z3_list[dist.get_rank()] = z3
        n1_list[dist.get_rank()] = n1

        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)
        z3 = torch.cat(z3_list, 0)
        n1 = torch.cat(n1_list, 0)

    #calculate sim
#######################################################################
    sim_z1_n1 = cls.sim(z1.unsqueeze(1), n1.unsqueeze(0))
    sim_z1_z2 = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    sim_LC = torch.cat([sim_z1_z2,sim_z1_n1],dim=-1)

    sim_z3_z1 = cls.sim(z3.unsqueeze(1), z1.unsqueeze(0))
    sim_z3_n1 = cls.sim(z3.unsqueeze(1), n1.unsqueeze(0))
    sim_LD = torch.cat([sim_z3_z1, sim_z3_n1], dim=-1)

    beta = 1
    total_loss = torch.zeros(batch_size)

    for i in range(batch_size):
        row_c = sim_LC[i]
        row_d = sim_LD[i]
        # 取出张量每一行中的第i个元素：z1和z2的相似度，以及其它元素：zi和zk的相似度
        # 计算LC
        ith_element_for_c = row_c[i].unsqueeze(0)
        exp_row_c = torch.exp(row_c)
        sum_exp_row_c = torch.sum(exp_row_c)

        pos_prob = torch.exp(ith_element_for_c) / sum_exp_row_c
        neg_prob = 1-pos_prob

        p_z1_z2_n1 = torch.log(pos_prob)
        p_zk_z2_n1 = torch.log(neg_prob)

        q = torch.ones_like(p_z1_z2_n1).to(cls.device)
        L_C =-(q * p_z1_z2_n1 + (1 - q) * p_zk_z2_n1)

        # 计算LD
        ith_element_for_d = row_d[i].unsqueeze(0)
        exp_row_d = torch.exp(row_d)
        sum_exp_row_d = torch.sum(exp_row_d)

        p_z1_z3_n1 = torch.log(torch.exp(ith_element_for_d) / sum_exp_row_d)
        p_zk_z3_n1 = torch.log(1 - torch.exp(ith_element_for_d) / sum_exp_row_d)

        L_D = -(pos_prob.detach() * p_z1_z3_n1 + neg_prob.detach() * p_zk_z3_n1)

        loss = L_C + beta * L_D
        total_loss[i] = loss

    average_loss = total_loss.mean().to(cls.device)



    cos_sim_all =torch.cat([sim_LC, sim_LD], dim=-1)

    if not return_dict:
        output = (cos_sim_all,) + outputs[2:]
        return ((average_loss,) + output) if average_loss is not None else output
    return SequenceClassifierOutput(
        loss=average_loss,
        logits=cos_sim_all,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
#######################################################################

def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    prompt_mask=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    batch_size = input_ids.shape[0]

    inputs_embeds = encoder.embeddings.word_embeddings(input_ids)

    ids = attention_mask.sum(1) - 3

    l = cls.endp.shape[0]

    for i in range(l):
        inputs_embeds[range(batch_size), ids - (l - i)] = cls.endp[i]

    outputs = encoder(
        None,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)

    if cls.pooler_type in ["cls", "mask"]  and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        prompt_mask=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                prompt_mask=prompt_mask,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                prompt_mask=prompt_mask,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )



class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        prompt_mask=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                prompt_mask=prompt_mask,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                prompt_mask=prompt_mask,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )
