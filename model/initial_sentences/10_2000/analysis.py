from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_data(data_path):
    text = pd.read_csv(data_path)
    initial = text['initial_sentence']

    offset = 6
    final = text['sentence'].str[offset:-offset]
    
    return initial, final

def get_loss(data, model):
    inp, mask = data.values()
    inp = torch.tensor(inp).unsqueeze(0).cuda()
    mask = torch.tensor(mask).unsqueeze(0).cuda()

    with torch.no_grad():
        loss = model(inp, mask, labels=inp)['loss'].item()
        return loss

def get_losses(model, initial, final):
    initial_losses, final_losses = [], []
    
    for item in tqdm(initial):
        initial_losses.append(get_loss(item, model))
    
    for item in tqdm(final):
        final_losses.append(get_loss(item, model))
            
    return initial_losses, final_losses

def average_over_batch(bsz, lst):
    return np.average(np.array(lst).reshape(-1, bsz), axis=1)

def main(model_name, data_path):

    initial, final = load_data(data_path)

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    def tokenize(example):
        return tokenizer(example['data'])

    initial = datasets.Dataset.from_dict({'data': initial.tolist()})
    final = datasets.Dataset.from_dict({'data': final.tolist()})

    initial = initial.map(tokenize, batched=True, remove_columns=initial.column_names)
    final = final.map(tokenize, batched=True, remove_columns=final.column_names)

    init_data, final_data = get_losses(model, initial, final)

    return average_over_batch(10, init_data), average_over_batch(10, final_data)
    
# def plot(init, final):
#     plt.plot(avg_init)
#     plt.plot(avg_final)
#     plt.xlabel("Iters")
#     plt.ylabel("CEL")
#     plt.savefig(f'{out_name}.png')
#     plt.show

# avg_init = average_over_batch(10, init_data)
# avg_final = average_over_batch(10, final_data)

# with open(f'{out_name}_data.npz', 'w') as f:
#     np.savez(f, init=avg_init, final=avg_final)

# plt.plot(avg_init)
# plt.plot(avg_final)
# plt.xlabel("Iters")
# plt.ylabel("CEL")
# plt.savefig(f'{out_name}.png')