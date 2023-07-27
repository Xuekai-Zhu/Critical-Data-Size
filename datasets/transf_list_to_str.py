import json

def convert_lists_to_strings(json_data):
    assert isinstance(json_data["output"], list)
    assert isinstance(json_data["instruction"], list)
    # assert len(json_data["output"]) == 1
    # assert len(json_data["instruction"]) == 1
    
    instruction = " ".join(json_data["instruction"])
    output = " ".join(json_data["output"])
    
    json_data["output"] = output  # Convert 'output' list to a JSON-encoded string

    json_data["instruction"] = instruction  # Convert 'instruction' list to a JSON-encoded string
    return json.dumps(json_data)

def process_json_file(input_file, output_file):
    with open(input_file, "r") as file:
        json_list = [json.loads(line.strip()) for line in file]

    processed_json_list = [convert_lists_to_strings(data) for data in json_list]

    with open(output_file, "w") as file:
        for data in processed_json_list:
            file.write(data + '\n')

# 使用示例
input_file = "./yesno_task/datatsets/old_version/test.json"  # 替换为您的输入JSON文件名
output_file = "./yesno_task/datatsets/test.json"  # 替换为您的输出JSON文件名
process_json_file(input_file, output_file)

# 使用示例
input_file = "./yesno_task/datatsets/old_version/train.json"  # 替换为您的输入JSON文件名
output_file = "./yesno_task/datatsets/train.json"  # 替换为您的输出JSON文件名
process_json_file(input_file, output_file)


# 使用示例
input_file = "./yesno_task/datatsets/old_version/valid.json"  # 替换为您的输入JSON文件名
output_file = "./yesno_task/datatsets/valid.json"  # 替换为您的输出JSON文件名
process_json_file(input_file, output_file)


# 使用示例
input_file = "./yesno_task/datatsets/old_version/pruning_set/train.json"  # 替换为您的输入JSON文件名
output_file = "./yesno_task/datatsets/train_pruning.json"  # 替换为您的输出JSON文件名
process_json_file(input_file, output_file)