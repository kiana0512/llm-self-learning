import json
from pathlib import Path
from sacrebleu import corpus_bleu

def load_data(pred_path, ref_path):
    with open(pred_path, 'r', encoding='utf-8') as f:
        predictions = [line.strip() for line in f if line.strip()]
    with open(ref_path, 'r', encoding='utf-8') as f:
        references = [json.loads(line)["response"].strip() for line in f if line.strip()]
    return predictions, references

def evaluate_bleu(pred_path, ref_path):
    preds, refs = load_data(pred_path, ref_path)
    assert len(preds) == len(refs), f"预测数：{len(preds)}，参考数：{len(refs)} 不一致"
    bleu = corpus_bleu(preds, [refs])
    print(f"🔍 BLEU分数: {bleu.score:.2f}")
    return bleu.score

if __name__ == "__main__":
    pred_path = "../data/preds.txt"
    ref_path = "../data/wmt19_zh_en.jsonl"
    evaluate_bleu(pred_path, ref_path)
