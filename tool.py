import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForMaskedLM, AutoTokenizer
from torch.nn import CosineSimilarity
from tqdm import tqdm
import torch.nn.functional as F
from torch.cuda.amp import autocast
import gc

def sim(x, y, temp=1.0):
    cos_sim = F.cosine_similarity(x, y, dim=-1)
    return cos_sim / temp

def prepare_features(sampled_sentences):
    # padding = longest (default)
    #   If no sentence in the batch exceed the max length, then use
    #   the max sentence length in the batch, otherwise use the
    #   max sentence length in the argument and truncate those that
    #   exceed the max length.
    # padding = max_length (when pad_to_max_length, for pressure test)
    #   All sentences are padded/truncated to data_args.max_seq_length.
    total = len(sampled_sentences)

    mask_token = tokenizer.mask_token
    pad_token = tokenizer.pad_token
    examples = {'sent0': [], 'sent1': [], 'sent2': [], 'neg_examples': [], 'prompts': [],'weak_pos_prompts':[], 'strong_pos_prompts':[],'neg_prompts':[]}

    # promptbert_prompts = [
    #                ["This sentence : \" ", " \" memans [MASK] .".replace('[MASK]', mask_token)],
    #                ["This sentence of \" ", " \" means [MASK] .".replace('[MASK]', mask_token)],
    #                ["This sentence : \' ", " \' memans [MASK] .".replace('[MASK]', mask_token)],
    #                ["The sentence : \' ", " \' means [MASK] .".replace('[MASK]', mask_token)],
    #                ]

    weak_pos_prompts = [
        ["Given \" ", " \" , we assume that \" [MASK] \"".replace('[MASK]', mask_token)],
        ["\" ", " \" , Is this review positive ? [MASK] .".replace('[MASK]', mask_token)],
        ["\" ", " \" , is related to \" [MASK] \"".replace('[MASK]', mask_token)],
        ["\" ", " \" is a [MASK] news".replace('[MASK]', mask_token)],
        ["\" ", " \" is a [MASK] one".replace('[MASK]', mask_token)],
        ["\" ", " \" . In summary : \" [MASK] \"".replace('[MASK]', mask_token)],
        ["Article \" ", " \" belongs to a [MASK] topic".replace('[MASK]', mask_token)],
        ["This sentence : \" ", " \" means [MASK] .".replace('[MASK]', mask_token)],
        ["By \" ", " \" they mean [MASK] .".replace('[MASK]', mask_token)],
    ]

    strong_pos_prompts = [
        ["Given \" ", " \" , it's crystal clear that \" [MASK] \"".replace('[MASK]', mask_token)],
        ["\" ", " \" , Is this review absolutely positive ? [MASK] .".replace('[MASK]', mask_token)],
        ["\" ", " \" is a vibrant testament to [MASK] .".replace('[MASK]', mask_token)],
        ["\" ", " \" is an absolutely specific expression of [MASK] .".replace('[MASK]', mask_token)],
        ["This sentence : \" ", " \" distinctly portrays : \" [MASK] \"".replace('[MASK]', mask_token)],
        ["\" ", " \" vividly highlights : \" [MASK] \"".replace('[MASK]', mask_token)],
        ["\" ", " \" , glaringly reflects [MASK] theme".replace('[MASK]', mask_token)],
        ["This sentence : \" ", " \" strongly emphasizes the [MASK] .".replace('[MASK]', mask_token)],
        ["By \" ", " \" , they undoubtedly means [MASK] .".replace('[MASK]', mask_token)],
    ]

    neg_prompts = [
        ["\" ", " \" , Is this review negative ? [MASK] .".replace('[MASK]', mask_token)],
        ["Without \" ", " \" they mean [MASK] .".replace('[MASK]', mask_token)],
        ["\" ", " \" is inconsistent to \" [MASK] \"".replace('[MASK]', mask_token)],
        ["\" ", " \" is totally different to : \" [MASK] \"".replace('[MASK]', mask_token)],
        ["\" ", " \" which not denotes [MASK] .".replace('[MASK]', mask_token)],
        ["\" ", " \" is not a [MASK] one".replace('[MASK]', mask_token)],
        ["This sentence : \" ", " \" not means [MASK] .".replace('[MASK]', mask_token)],
        ["Article \" ", " \" is definitely not about the [MASK] topic".replace('[MASK]', mask_token)],

    ]

    # _prompt = ["", " [PAD] [PAD] [PAD] [PAD] [MASK] .".replace('[PAD]', pad_token).replace('[MASK]', mask_token)]
    # _prompt = promptbert_prompts[model_args.prompt_id]

    num = prompt_len
    if plain_bert is True:
        _prompt =["\' ", " \'".replace('[MASK]', mask_token)]
    else:
        _prompt = ["", " ".join([" "] + ["[PAD]" for _ in range(num)] + ["[MASK] ."]).replace('[PAD]', pad_token).replace(
        '[MASK]', mask_token)]

    # one_third = total // 3
    # two_thirds = 2 * total // 3
    # Avoid "None" fields
    for idx in range(total):
        sent = sampled_sentences[idx]

        examples['sent0'].append(sent)
        examples['prompts'].append(_prompt)

        nid = random.randint(0, len(weak_pos_prompts) - 1)
        examples['weak_pos_prompts'].append(weak_pos_prompts[nid])
        examples['sent1'].append(sent)

        nid = random.randint(0, len(strong_pos_prompts) - 1)
        examples['strong_pos_prompts'].append(strong_pos_prompts[nid])
        examples['sent2'].append(sent)

        nid = random.randint(0, len(neg_prompts) - 1)
        examples['neg_examples'].append(sent)
        examples['neg_prompts'].append(neg_prompts[nid])

    sentences = examples['sent0'] + examples['sent1'] + examples['sent2'] + examples['neg_examples']

    prompts = examples['prompts'] + examples['weak_pos_prompts'] + examples['strong_pos_prompts'] + examples[
        'neg_prompts']

    sent_features = {'input_ids': [], 'attention_mask': [], 'prompt_mask': []}

    for i, s in enumerate(sentences):
        s = tokenizer.encode(s, add_special_tokens=False)[:max_seq_length]

        pre = tokenizer.encode(prompts[i][0])[:-1]
        end = tokenizer.encode(prompts[i][1])[1:]
        sent_features['input_ids'].append(pre + s + end)
        sent_features['prompt_mask'].append([0] + (len(pre) - 1) * [1] + len(s) * [0] + (len(end) - 1) * [1] + [0])

    max_len = max([len(s) for s in sent_features['input_ids']])
    for i in range(len(sent_features['input_ids'])):
        s = sent_features['input_ids'][i]
        ls = len(s)
        sent_features['input_ids'][i] = s + [tokenizer.pad_token_id] * (max_len - ls)
        sent_features['attention_mask'].append(ls * [1] + (max_len - ls) * [0])
        sent_features['prompt_mask'][i] = sent_features['prompt_mask'][i] + (max_len - ls) * [0]

    features = {}

    for key in sent_features:
        features[key] = [[sent_features[key][i],
                          sent_features[key][i + total],
                          sent_features[key][i + total * 2],
                          sent_features[key][i + total * 3]
                          ] for i in range(total)]

    return features


def batchify_features(features, batch_size):
    # 确定有多少批次
    num_batches = len(features['input_ids']) // batch_size + (0 if len(features['input_ids']) % batch_size == 0 else 1)
    # 创建一个列表来存放批次数据
    batched_features = []

    for i in range(num_batches):
        # 计算当前批次的开始和结束索引
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        # 获取每个批次的数据
        batch = {key: value[start_idx:end_idx] for key, value in features.items()}

        # 添加当前批次到列表中
        batched_features.append(batch)

    return batched_features

def pooler(attention_mask,outputs):
     last_hidden = outputs.hidden_states[-1]
     ids = attention_mask.sum(1) - 3
     return last_hidden[range(last_hidden.shape[0]), ids].cpu()

def get_cls_embedding(attention_mask, outputs):
    last_hidden = outputs.hidden_states[-1]
    return last_hidden[range(last_hidden.shape[0]), 0].cpu()

if __name__=="__main__":
    # 配置参数
    model_name_or_path = "result/CLSESSP-bb:seed42"  # 模型路径
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    batch_size = 128  # 根据您的硬件配置调整
    max_seq_length = 32
    num_samples = 2048  # 从wiki文件中随机抽取的样本数
    wiki_path = "data/wiki1m_for_simcse.txt"  # 更改为您的wiki文件路径
    cos_sim = CosineSimilarity(dim=0)
    temperature = 0.05  # 相似度计算的温度参数
    mask_token = tokenizer.mask_token
    pad_token = tokenizer.pad_token
    prompt_len = 4
    plain_bert = False

    # 从wiki文件中读取数据
    with open(wiki_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        sampled_sentences = random.sample(lines, num_samples)


    if plain_bert is not True:
        state_dict = torch.load(model_name_or_path+'/pytorch_model.bin')
        endp = state_dict['endp']
        endp = endp.to(device)

    weak_pos_probs = []
    weak_neg_probs = []
    strong_pos_probs = []
    strong_neg_probs = []

    batched_features = batchify_features(prepare_features(sampled_sentences), batch_size)

    for batch in batched_features:
        input_ids = torch.tensor(batch['input_ids']).to(device)
        attention_mask = torch.tensor(batch['attention_mask']).to(device)
        prompt_mask = torch.tensor(batch['prompt_mask']).to(device)

        batch_size = input_ids.size(0)
        # Number of sentences in one instance
        num_sent = input_ids.size(1)

        mlm_outputs = None

        inputs_embeds = model.bert.embeddings.word_embeddings(input_ids)
        inputs_embeds_s0 = inputs_embeds[:, 0]
        inputs_embeds_s1 = inputs_embeds[:, 1:]
        attention_mask_s0 = attention_mask[:, 0]
        prompt_mask_s0 = prompt_mask[:, 0]

        ids = attention_mask_s0.sum(1) - 3
        if plain_bert is not True:
            l = endp.shape[0]

            for i in range(l):
                inputs_embeds_s0[range(batch_size), ids - (l - i)] = endp[i]

            inputs_embeds = torch.cat([inputs_embeds_s0.unsqueeze(1),
                                       inputs_embeds_s1], dim=1)

        # Flatten input for encoding
        input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
        inputs_embeds = inputs_embeds.view((batch_size * num_sent, -1, inputs_embeds.size(-1)))
        attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (bs * num_sent len)
        prompt_mask = prompt_mask.view((-1, prompt_mask.size(-1)))  # (bs * num_sent len)

        with autocast():
            with torch.no_grad():
              # Get raw embeddings
              outputs = model(
                  None,
                  attention_mask=attention_mask,
                  inputs_embeds=inputs_embeds,
                  output_hidden_states=True
                )

              p_outputs = model(
                  None,
                  attention_mask=prompt_mask,
                  inputs_embeds=inputs_embeds,
                  output_hidden_states=True
                )

            # Pooling
            pooler_output = pooler(attention_mask, outputs)
            pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1)))  # (bs, num_sent, hidden)

            p_pooler_output = pooler(attention_mask, p_outputs)
            p_pooler_output = p_pooler_output.view((batch_size, num_sent, p_pooler_output.size(-1)))  # (bs, num_sent, hidden)

            pooler_output = pooler_output - p_pooler_output

            # Separate representation
            z1, z2, z3, n1 = pooler_output[:, 0], pooler_output[:, 1], pooler_output[:, 2], pooler_output[:, 3]

            if plain_bert is True:
                z1_output = get_cls_embedding(attention_mask, outputs)
                z1_output = z1_output.view((batch_size, num_sent, pooler_output.size(-1)))
                z1 = z1_output[:, 0]

            sim_z1_n1 = sim(z1.unsqueeze(1), n1.unsqueeze(0))
            sim_z1_z2 = sim(z1.unsqueeze(1), z2.unsqueeze(0))
            sim_LC = torch.cat([sim_z1_z2, sim_z1_n1], dim=-1)

            sim_z3_z1 = sim(z3.unsqueeze(1), z1.unsqueeze(0))
            sim_z3_n1 = sim(z3.unsqueeze(1), n1.unsqueeze(0))
            sim_LD = torch.cat([sim_z3_z1, sim_z3_n1], dim=-1)

            beta = 1
            total_loss = torch.zeros(batch_size)

            for i in range(batch_size):
                row_c = sim_LC[i]
                row_d = sim_LD[i]
                # 取出张量每一行中的第i个元素：z1和z2的相似度，以及其它元素：zi和zk的相似度
                ith_element_for_c = row_c[i].unsqueeze(0)
                exp_row_c = torch.exp(row_c)
                sum_exp_row_c = torch.sum(exp_row_c)

                pos_prob = torch.exp(ith_element_for_c) / sum_exp_row_c
                neg_prob = 1 - pos_prob

                p_z1_z2_n1 = pos_prob
                p_zk_z2_n1 = neg_prob

                ith_element_for_d = row_d[i].unsqueeze(0)
                exp_row_d = torch.exp(row_d)
                sum_exp_row_d = torch.sum(exp_row_d)

                p_z1_z3_n1 = torch.exp(ith_element_for_d) / sum_exp_row_d
                p_zk_z3_n1 = 1 - torch.exp(ith_element_for_d) / sum_exp_row_d

                weak_pos_probs.append(p_z1_z2_n1)
                weak_neg_probs.append(p_zk_z2_n1)
                strong_pos_probs.append(p_z1_z3_n1)
                strong_neg_probs.append(p_zk_z3_n1)

                # 在这里释放变量和调用垃圾收集
                torch.cuda.empty_cache()  # 清空CUDA缓存
                gc.collect()  # 显式调用垃圾收集器

    weak_pos_probs = torch.tensor([t.item() for t in weak_pos_probs])
    weak_neg_probs = torch.tensor([t.item() for t in weak_neg_probs])
    strong_pos_probs = torch.tensor([t.item() for t in strong_pos_probs])
    strong_neg_probs = torch.tensor([t.item() for t in strong_neg_probs])

    weak_pos_probs = weak_pos_probs.numpy()
    weak_neg_probs = weak_neg_probs.numpy()
    strong_pos_probs = strong_pos_probs.numpy()
    strong_neg_probs = strong_neg_probs.numpy()

    # 设置直方图的bins大小
    bins = np.linspace(0, 0.02, 1000)  # 增加bins的数量

    # 绘制直方图
    plt.figure(figsize=(10, 6))

    # 绘制"弱"概率分布
    plt.hist(weak_pos_probs, bins=bins, alpha=0.7, label='Weak Positive', color='skyblue')
    # 绘制"强"概率分布
    plt.hist(strong_pos_probs, bins=bins, alpha=0.7, label='Strong Positive', color='orange')

    # 添加图例
    plt.legend(loc='upper left', fontsize=20)

    # 添加标题和轴标签
    plt.title('Probability Distributions of Positive Pairs', fontsize=20)
    plt.xlabel('Probability', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)

    # 设置边界和格网
    plt.grid(False)
    plt.xlim(0.005, 0.012)
    plt.ylim(0, plt.ylim()[1])  # 自动获取y轴的上限


    # 移除顶部和右侧的轴线
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # 调整轴线上数字的大小
    plt.tick_params(axis='both', which='major', labelsize=16)

    # 保存图像
    plt.savefig('result/positive_pairs_distribution.png', dpi=300, bbox_inches='tight')
    print("save success!")
    # 关闭plt，释放内存
    plt.close()

