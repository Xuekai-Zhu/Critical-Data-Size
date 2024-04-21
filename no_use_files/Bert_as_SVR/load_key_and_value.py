import torch
import numpy as np
import os
from tqdm import tqdm
from transformers import BertForSequenceClassification
import torch.nn.utils.rnn as rnn_utils

def load_file(input_file, return_list=False):
    loaded_arrays_list = []
    # 使用 np.load 加载数据
    with np.load(input_file) as data:
        if return_list:
            for key in tqdm(data.keys()):
                loaded_arrays_list.append(data[key])
            return loaded_arrays_list
        else:
            return data

def standardization(data, axis=None):
    mean_val = np.mean(data, axis=axis, keepdims=True)
    std_val = np.std(data, axis=axis, keepdims=True)

    normalized_result = (data - mean_val) / std_val
    return normalized_result


def classifier(input_hidden):
    model = BertForSequenceClassification.from_pretrained(
        "model/bert-base-uncased",
        local_files_only=True,
        num_labels=2, 
        # problem_type="multi_label_classification"
        # cache_dir=training_args.cache_dir,
    )  
    model.config.use_cache = False
    # print(model)
    model.eval()
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    project_v = model.classifier(input_hidden.to(device)).detach().cpu().numpy()
    
    return project_v
    

def Calculate_Lagrange_multipliers(key_file, vaule_file, save_file):
    # key = load_file(key_file)
    key = np.load(key_file)
    vaule = np.load(vaule_file)
    # length = len(key)
    # assert len(key) == len(vaule)
    final_results = []
    for index in tqdm(key.keys()):
        k = key[index]
        v = vaule[index]
        
        N, D = k.shape
        
        partial_result = np.dot(k, k.T) / np.sqrt(D)
        stand_results = standardization(partial_result, axis=-1)
        
        exp_result = np.exp(stand_results)
        sum_exp_result = np.sum(exp_result, axis=-1, keepdims=True)
        # mean_exp_result = np.mean(exp_result, axis=-1, keepdims=True)
        results = v * sum_exp_result
        
        # project_results = classifier(results)
        final_results.append(results)

    np.savez(save_file, *final_results)

def Calculate_project_multipliers(in_file, save_file):
    multipliers = np.load(in_file)
    batch = []
    bs = 64
    final_results = []
    for i, index in enumerate(tqdm(multipliers.keys())):
        if len(batch) < bs:
            mul = multipliers[index]
            batch.append(torch.tensor(mul).squeeze(0))
        else:
            lengths = [len(seq) for seq in batch]
            padded_seqs = rnn_utils.pad_sequence(batch, batch_first=True)
            mask_attention = torch.zeros(padded_seqs.size(0), padded_seqs.size(1))
            for j, length in enumerate(lengths):
                mask_attention[j, :length] = 1
            
            project_results = classifier(padded_seqs)
            for b in range(padded_seqs.shape[0]):
                mask = mask_attention[b]
                project_result = project_results[b]
                extracted_sequence = project_result[mask == 1]
                final_results.append(extracted_sequence)
            
            batch = []
            mul = multipliers[index]
            batch.append(torch.tensor(mul).squeeze(0))
            
                
    np.savez(save_file, *final_results)



if __name__ == '__main__':
    # first half
    # key_file = "model/bert-base-uncased/lagrange_multiplier/intermediate_key_0.5.npz"
    # vaule_file = "model/bert-base-uncased/lagrange_multiplier/intermediate_vaule_0.5.npz"
    # save_file = "model/bert-base-uncased/lagrange_multiplier/final_multiplier_0.5.npz"
    # Calculate_Lagrange_multipliers(key_file, vaule_file, save_file)
    
    # save_project_file = "model/bert-base-uncased/lagrange_multiplier/project_final_multiplier_0.5.npz"
    # # Calculate_project_multipliers(save_file, save_project_file)
    
    # second half
    key_file = "model/bert-base-uncased/lagrange_multiplier/intermediate_key.npz"
    vaule_file = "model/bert-base-uncased/lagrange_multiplier/intermediate_vaule.npz"
    save_file = "model/bert-base-uncased/lagrange_multiplier/final_multiplier.npz"
    Calculate_Lagrange_multipliers(key_file, vaule_file, save_file)
    
    save_project_file = "model/bert-base-uncased/lagrange_multiplier/project_final_multiplier.npz"
    Calculate_project_multipliers(save_file, save_project_file)
    