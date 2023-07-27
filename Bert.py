from transformers import BertModel, BertTokenizer
from sklearn import svm
import torch
from data_process import Datasets
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm



def save2vector(input_file, save_data, save_target):

    # 加载数据集
    train_set = Datasets(input_file)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=False)

    # 加载预训练的BERT模型和分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir="./pre-trained-model/")
    bert_model = BertModel.from_pretrained('bert-base-uncased', cache_dir="./pre-trained-model/").cuda()

    # 使用 BERT 编码训练数据
    train_features = []
    train_labels = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            # if batch_idx > 10:
            #     break
            text = batch['text']  # 假设文本数据的键为'text'
            target = batch['target']  # 假设标签数据的键为'target'
            # input_ids = batch['input_ids']
            # labels = batch['label']
            labels = np.where(np.array(target) == 'yes', 1, 0)
                
            inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)
            inputs = {k: v.cuda() for k, v in inputs.items()}  # 将输入移动到GPU
            outputs = bert_model(**inputs)
            features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            train_features.append(features)
            train_labels.append(labels)
            

    # # 使用BERT模型获取句子表示
    # with torch.no_grad():
    #     outputs = model(**inputs)
    #     representations = outputs.last_hidden_state[:, 0, :].numpy()

    # # 使用SVM训练这些表示
    # clf = svm.SVC()
    # clf.fit(representations, labels)

    train_features = np.concatenate(train_features)
    train_labels = np.concatenate(train_labels)


    np.save(save_data, train_features)
    np.save(save_target, train_labels)


    # load

    # train_features = np.load("./datasets/natural-instructions-2.8/yesno_task/datatsets/vector_data_from_bert/train_features.npy")
    # train_labels = np.load("./datasets/natural-instructions-2.8/yesno_task/datatsets/vector_data_from_bert/train_labels.npy")
    # print(train_features)
    # print(train_labels)
if __name__ == '__main__':
    # train
    train_path = "./datasets/natural-instructions-2.8/yesno_task/datatsets/train.json"
    save_data = "./datasets/natural-instructions-2.8/yesno_task/datatsets/vector_data_from_bert/train_features.npy"
    save_target = save_data.replace("features", "labels")
    save2vector(train_path, save_data, save_target)
    
    # vaild
    vaild_path = "./datasets/natural-instructions-2.8/yesno_task/datatsets/valid.json"
    save_data = "./datasets/natural-instructions-2.8/yesno_task/datatsets/vector_data_from_bert/valid_features.npy"
    save_target = save_data.replace("features", "labels")
    save2vector(vaild_path, save_data, save_target)
    
    # test
    test_path = "./datasets/natural-instructions-2.8/yesno_task/datatsets/test.json"
    save_data = "./datasets/natural-instructions-2.8/yesno_task/datatsets/vector_data_from_bert/test_features.npy"
    save_target = save_data.replace("features", "labels")
    save2vector(test_path, save_data, save_target)
    