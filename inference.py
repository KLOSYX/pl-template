import json
import itertools
import re
import time
import importlib
import sys
from pathlib2 import Path
sys.path.append(str(Path(__file__).parent.absolute()))

import torch
from tqdm import tqdm

from utils.processor import Processor


MODEL_NAME = "gaiic_visualbert_classifier"
TOKENIZER_NAME = "bert-base-chinese"


def load_model():
    camel_name = "".join([i.capitalize() for i in MODEL_NAME.split("_")])
    try:
        model = getattr(
            importlib.import_module(
                "model." + MODEL_NAME, package=__package__),
            camel_name,
        )
    except Exception:
        raise ValueError(
            f"Invalid Module File Name or Invalid Class Name {MODEL_NAME}.{camel_name}!"
        )
    return model

def get_attr_dict(attr_to_attrvals_file):
    attr_dict = {}
    with open(attr_to_attrvals_file, "r") as f:
        for attr, attrval_list in json.load(f).items():
            attrval_list = list(map(lambda x: x.split("="), attrval_list))
            attr_dict[attr] = list(itertools.chain.from_iterable(attrval_list))
    return attr_dict


def match_attrval(attr_dict, title, attr):
    # 在title中匹配属性值
    attrvals = "|".join(attr_dict[attr])
    ret = re.findall(attrvals, title)
    return "{}{}".format(attr, "".join(ret))


def initialize_model(checkpoint_path):
    processor = Processor(TOKENIZER_NAME)
    # model = gaiic_bert_classifer.GaiicBertClassifer(
    #     MODEL_NAME, processor.get_vocab_size(), processor.get_padding_idx(), 0
    # )
    model = load_model().load_from_checkpoint(checkpoint_path)
    model.cuda()
    # checkpoint = torch.load(checkpoint_path, map_location="cuda")
    # sd = {}
    # for k, v in checkpoint['state_dict'].items():
    #     k = k.replace("module.", "")
    #     sd[k] = v
    # model.load_state_dict(sd)

    # for p in model.parameters():
    #     p.data = p.data.float()
    #     if p.grad:
    #         p.grad.data = p.grad.data.float()

    return processor, model.eval()


def predict_one(processor, model, text, feature, logits=False):
    if isinstance(feature, list):
        feature = torch.tensor(feature, dtype=torch.float)
    feature = feature.unsqueeze(0)
    encodings = processor(feature, text)
    encodings = [x.cuda() for x in encodings]
    outputs = model(*encodings)
    if logits:
        return outputs.tolist()
    predicts = torch.argmax(outputs, dim=1)
    return predicts.item()


def main(logits=False):
    checkpoint = "lightning_logs/gaiic2022-visualbert-base-nni/2c0zwlxv/checkpoints/best-epoch=32-val_loss=0.067-val_acc=0.9430.ckpt"
    test_file = "../train/preliminary_testB.txt"
    output_file = f"predicts/testA_predict_{time.time()}.txt"
    attr_to_attrvals_file = "../train/attr_to_attrvals.json"
    attr_dict = get_attr_dict(attr_to_attrvals_file)

    processor, model = initialize_model(checkpoint)

    outputs = []
    with open(test_file, "r", encoding="utf-8") as infile:
        for line in tqdm(infile):
            predicts = {}
            match = {}
            item = json.loads(line)
            feature = item["feature"]
            title = item["title"]
            predicts["img_name"] = item["img_name"]
            match["图文"] = predict_one(processor, model, title, feature, logits)
            if item["query"]:
                for q in item["query"][1:]:
                    match[q] = predict_one(
                        processor, model, match_attrval(attr_dict, title, q), feature, logits
                    )
            predicts["match"] = match
            outputs.append(json.dumps(predicts, ensure_ascii=False) + "\n")

    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.writelines(outputs)


if __name__ == "__main__":
    main(False)

