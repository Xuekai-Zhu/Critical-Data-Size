import json
import os


one_layer_bert_path = "model-wise_grokking/experiments_config/one-layer-bert/config.json"

with open(one_layer_bert_path, "r") as f:
    config_dict = json.loads(f.read())
    
hidden_sizes = range(16, 256, 16)

for h in hidden_sizes:
    new_dir_names = os.path.join("model-wise_grokking/experiments_config", f"hidden_size_{h}")
    if not os.path.exists(new_dir_names):
        os.makedirs(new_dir_names)
        
    new_file = os.path.join(new_dir_names, "config.json")
    
    config_dict["hidden_size"] = h
    with open(new_file, "w") as f:
        f.write(json.dumps(config_dict, indent=4))