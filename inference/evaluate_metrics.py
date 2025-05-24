import json
from pathlib import Path
from sacrebleu import corpus_bleu, corpus_chrf
from rouge_score import rouge_scorer
from bert_score import score as bert_score

def load_data(pred_path, ref_path):
    with open(pred_path, 'r', encoding='utf-8') as f:
        predictions = [line.strip() for line in f if line.strip()]
    with open(ref_path, 'r', encoding='utf-8') as f:
        references = [json.loads(line)["response"].strip() for line in f if line.strip()]
    assert len(predictions) == len(references), f"数量不一致：预测 {len(predictions)}，参考 {len(references)}"
    return predictions, references

def evaluate_all_metrics(preds, refs):
    print("🔍 评估指标如下：")

    # BLEU
    bleu = corpus_bleu(preds, [refs])
    print(f"BLEU: {bleu.score:.2f}")

    # chrF
    chrf = corpus_chrf(preds, [refs])
    print(f"chrF: {chrf.score:.2f}")

    # ROUGE
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [rouge.score(r, p) for r, p in zip(refs, preds)]
    avg_rouge1 = sum(s["rouge1"].fmeasure for s in scores) / len(scores)
    avg_rouge2 = sum(s["rouge2"].fmeasure for s in scores) / len(scores)
    avg_rougeL = sum(s["rougeL"].fmeasure for s in scores) / len(scores)
    print(f"ROUGE-1: {avg_rouge1*100:.2f}")
    print(f"ROUGE-2: {avg_rouge2*100:.2f}")
    print(f"ROUGE-L: {avg_rougeL*100:.2f}")

    # BERTScore
    print("（BERTScore 计算中，可能需要几分钟）")
    P, R, F1 = bert_score(preds, refs, lang='zh', verbose=True)
    print(f"BERTScore-F1: {F1.mean().item()*100:.2f}")

if __name__ == "__main__":
    pred_path = "../data/preds.txt"
    ref_path = "../data/wmt19_zh_en.jsonl"
    preds, refs = load_data(pred_path, ref_path)
    evaluate_all_metrics(preds, refs)
