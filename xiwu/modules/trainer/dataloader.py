
from typing import Dict, List, Tuple
import os, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import json
import transformers

from xiwu.apis.fastchat_api import (
    LazySupervisedDataset,
    SupervisedDataset,
    rank0_print,
    preprocess,
    SeparatorStyle,
)
from transformers.trainer_pt_utils import LabelSmoother
from xiwu.configs import DataArgs
from xiwu.apis.xiwu_api import BaseQADatasetSaver
from xiwu import ASSEMBLER

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

class XLazySupervisedDataset(LazySupervisedDataset):
    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__(raw_data, tokenizer)


    def one_entry2conversation(self, one_entry):
        if 'conversations' in one_entry:
            return one_entry['conversations']
        elif 'conversation' in one_entry:
            return one_entry['conversation']
        else:  # 一轮对话
            q = one_entry['question']
            a = one_entry['answer']
            system_msg = one_entry.get('misc', {}).get('system_message', None)
            if system_msg:  # TODO, 这里处理似乎不太对
                q = f'{system_msg}\n\n{q}'
            conv = [
                {'from': 'human', 'value': q},
                {'from': 'gpt', 'value': a},
            ]
            return conv
            
        
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]
        # 加载xiwu-format格式的数据
        conv = self.one_entry2conversation(self.raw_data[i])

        # ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer)
        ret = self.preprocess([conv], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret
    
    @classmethod
    def preprocess(cls, sources,
            tokenizer: transformers.PreTrainedTokenizer,
        ) -> Dict:
        conv = ASSEMBLER.get_conversation_template('xiwu')
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        # Tokenize conversations
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        targets = input_ids.clone()

        assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

        # Mask targets. Only compute loss on the assistant outputs.
        sep = conv.sep + conv.roles[1] + ": "
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            turns = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_TOKEN_ID
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                if i != 0 and not tokenizer.legacy:
                    # The legacy and non-legacy modes handle special tokens differently
                    instruction_len -= 1

                # Ignore the user instructions
                target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
                cur_len += turn_len

                if i != 0 and not tokenizer.legacy:
                    # The legacy and non-legacy modes handle special tokens differently
                    cur_len -= 1

            target[cur_len:] = IGNORE_TOKEN_ID

            if False:  # Inspect and check the correctness of masking
                z = target.clone()
                z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
                rank0_print(tokenizer.decode(z))
                exit()

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_TOKEN_ID
                    rank0_print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" #turn = {len(turns) - 1}. (ignored)"
                    )

        return dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )


class DataAssembler:
    def __init__(self) -> None:
        pass

    @classmethod
    def get_train_eval_test_data(cls, data_files: List[str]) -> Tuple:
        train_raw_data = []
        eval_raw_data = []
        test_raw_data = []

        max_len_file_name = max([len(Path(file).name) for file in data_files])

        for i, file in enumerate(data_files):
            # rank0_print(f"Loading data file [{i+1}/{len(data_files)}]: {Path(file).name:<{max_len_file_name}}...")
            with open(file, "r") as f:
                data = json.load(f)
                for entry in data["entities"]:
                    if entry["misc"]["used_for"] == "train":
                        train_raw_data.append(entry)
                    elif entry["misc"]["used_for"] == "eval":
                        eval_raw_data.append(entry)
                    elif entry["misc"]["used_for"] == "test":
                        test_raw_data.append(entry)
                    else:
                        raise ValueError(f"Unknown split type: {entry['misc']['used_for']}")
        # rank0_print()
        rank0_print(f"Dataloaded, train: {len(train_raw_data)}, eval: {len(eval_raw_data)}, test: {len(test_raw_data)}")
        return train_raw_data, eval_raw_data, test_raw_data

    @classmethod
    def make_supervised_data_module(cls,
        tokenizer,
        data_args: DataArgs,
        
    ) -> Dict:
        
        data_path = data_args.data_path
        eval_data_path = data_args.eval_data_path

        if os.path.isfile(data_path):
            data_files = [data_path]
        else:
            data_files = []
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if file.endswith(".json"):
                        data_files.append(os.path.join(root, file))
        # 获取train_json
        train_raw_data, eval_raw_data, test_raw_data = DataAssembler.get_train_eval_test_data(data_files)

        dataset_cls = (XLazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset)
        train_dataset = dataset_cls(train_raw_data, tokenizer=tokenizer)
        eval_dataset = dataset_cls(eval_raw_data, tokenizer=tokenizer)
        test_dataset = dataset_cls(test_raw_data, tokenizer=tokenizer)
        
        return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)
        # return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, test_dataset=test_dataset)


