import json
import re
import os

def load_json_file(in_file, return_dict=False):
    with open(in_file, "r") as f:
        data = f.readlines()
    if return_dict:
        all_data = [json.loads(i) for i in data]
        return all_data
    else:
        return data
    
def save_answer_and_label(predicts, labels, save_dir):
    
    save_file = os.path.join(save_dir, "results.json")
    with open(save_file, "w") as f:
        for i, j in zip(predicts, labels):
            reults = {"predict":i, "label":j}
            item = json.dumps(reults)
            f.write(item + "\n")
    
    
def extract_output(predicts):
    results = []
    for i in predicts:
        match = re.search(r'Answer:\s*(\S*)', i)
        answer = match.group(1) if match else None
        results.append(answer)
    
    return results

def extract_binary_classifier(predicts):
    results = []
    for i in predicts:
        if i == "1":
            results.append("yes")
        elif i == "0":
            results.append("no")
    return results

def eval(in_file, label_file):
    save_dir = in_file.replace("/generated_predictions.json", "")
    save_dir = os.path.join(save_dir, "eval_results")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    predicts_json = load_json_file(in_file, return_dict=True)
    predicts = [i["output"] for i in predicts_json]
    predicts= extract_output(predicts)
    
    # count none
    none_count = predicts.count(None)
    print(f"The number of None values in the list is: {none_count}")
    
    # Find the indices of None values
    none_indices = [i for i, x in enumerate(predicts) if x is None]
    print(f"The indices of None values are: {none_indices}")
    
    labels_json = load_json_file(label_file, return_dict=True)
    labels = [i["output"] for i in labels_json][:len(predicts)]
    
    # save results
    save_answer_and_label(predicts, labels, save_dir)
    
    # Check if the lists are of the same length
    if len(predicts) != len(labels):
        print("The prediction and label lists are not of the same length.")
        return

    # Calculate the accuracy
    correct_predictions = sum(p == l for p, l in zip(predicts, labels))
    accuracy = correct_predictions / len(predicts)

    print(f"The accuracy is: {accuracy * 100}%")
    
    # Save the results to a file in the save_dir directory
    with open(os.path.join(save_dir, 'accuracy_results.txt'), 'w') as f:
        f.write(f"The accuracy is: {accuracy * 100}%" + "\n")
        f.write(f"The number of None values in the list is: {none_count}" + "\n")
        f.write(f"The indices of None values are: {none_indices}" + "\n")
    
    
def eval_binary_classifier(in_file, label_file):
    save_dir = in_file.replace("/generated_predictions.json", "")
    save_dir = os.path.join(save_dir, "eval_results")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    predicts_json = load_json_file(in_file, return_dict=True)
    predicts = [i["output"] for i in predicts_json]
    predicts= extract_binary_classifier(predicts)
    
    # count none
    none_count = predicts.count(None)
    print(f"The number of None values in the list is: {none_count}")
    
    # Find the indices of None values
    none_indices = [i for i, x in enumerate(predicts) if x is None]
    print(f"The indices of None values are: {none_indices}")
    
    labels_json = load_json_file(label_file, return_dict=True)
    labels = [i["output"] for i in labels_json][:len(predicts)]
    
    # save results
    save_answer_and_label(predicts, labels, save_dir)
    
    # Check if the lists are of the same length
    if len(predicts) != len(labels):
        print("The prediction and label lists are not of the same length.")
        return

    # Calculate the accuracy
    correct_predictions = sum(p == l for p, l in zip(predicts, labels))
    accuracy = correct_predictions / len(predicts)

    print(f"The accuracy is: {accuracy * 100}%")
    
    class_0_right = 0
    class_0_num = 0
    class_1_right = 0
    class_1_num = 0
    
    # separate the calss
    for p, l in zip(predicts, labels):
        if l == "no":
            class_0_num += 1
            if p == l:
               class_0_right += 1
        elif l == "yes":
            class_1_num += 1
            if p == l:
               class_1_right += 1
               
            
    class_0_acc = class_0_right / class_0_num
    class_1_acc = class_1_right / class_1_num
    
    print(f"The class_0 accuracy is: {class_0_acc * 100}%")
    print(f"The class_1 accuracy is: {class_1_acc * 100}%")
    
    # Save the results to a file in the save_dir directory
    with open(os.path.join(save_dir, 'accuracy_results.txt'), 'w') as f:
        f.write(f"The accuracy is: {accuracy * 100}%" + "\n")
        f.write(f"The class_0 accuracy is: {class_0_acc * 100}%" + "\n")
        f.write(f"The class_1 accuracy is: {class_1_acc * 100}%" + "\n")
        f.write(f"The number of None values in the list is: {none_count}" + "\n")
        f.write(f"The indices of None values are: {none_indices}" + "\n")

if __name__ == '__main__':
    # eval decoder model
    # label_file = "datasets/yesno_data/test.json"
    # predict_file= "model/opt-1.3b-from-pretrain/opt-1.3b-equal-tasks-random-0.4-pruning" + "/generated_predictions.json"
    # eval(predict_file, label_file)
    
    # eval encoder model
    # label_file = "datasets/yesno_task/datatsets/test.json"
    # predict_file= "model/bert-base-SVR_pruning/multip_class/30%" + "/generated_predictions.json"
    # eval_binary_classifier(predict_file, label_file)
    
    # eval on 0.18 size test
    label_file = "datasets/yesno_task/random_pruning_on_testsets/0.18_test.json"
    predict_file= "model/bert-base-from-config/0.18_test" + "/generated_predictions.json"
    eval_binary_classifier(predict_file, label_file)
    
    
    # eval on task 66
    # label_file = "datasets/yesno_task/task066/test.json"
    # predict_file= "model/random_pruning_on_task/bert-base-0.1-random/task066" + "/generated_predictions.json"
    # eval_binary_classifier(predict_file, label_file)
    