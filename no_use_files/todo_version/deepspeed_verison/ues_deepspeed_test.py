from transformers import BertModel
from deepspeed.pipe import PipelineModule
import deepspeed


from transformers import AutoModel
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live

model = AutoModel.from_pretrained("pre-trained-model/huggyllama/llama-7b")

# estimate_zero2_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)