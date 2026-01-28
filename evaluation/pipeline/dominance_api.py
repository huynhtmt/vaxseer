# VaxSeer/evaluation/pipeline/dominance_api.py
import os, csv
from typing import List, Tuple
from Bio import SeqIO

def _read_fasta(path: str) -> List[Tuple[str, str]]:
    recs = []
    for rec in SeqIO.parse(path, "fasta"):
        recs.append((rec.id, str(rec.seq)))
    return recs

def run_dominance(cand_fasta: str,
                  circ_fasta: str,
                  lm_ckpt: str,
                  out_csv: str,
                  mode: str = "auto") -> str:
    """
    Dominance API.
    If lm_ckpt is a valid file (or mode='flu_lm'), run the real model.
    Otherwise, fallback to the placeholder (score = len(sequence)).
    """
    out_dir = os.path.dirname(out_csv)
    if out_dir: os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(cand_fasta):
        raise FileNotFoundError(f"[run_dominance] Candidate FASTA not found: {cand_fasta}")

    cands = _read_fasta(cand_fasta)
    if not cands:
        raise RuntimeError(f"[run_dominance] No candidates in {cand_fasta}")

    rows = []

    # Decide which branch to use
    use_real = bool(lm_ckpt) and os.path.isfile(lm_ckpt)
    if mode == "placeholder": use_real = False
    if mode == "flu_lm":      use_real = True

    if use_real:
        try:
            import torch
            # Try your project's LM class (adjust paths/names as needed)
            LM = None
            try:
                # Example: VaxSeer Lightning model class
                from vaxseer.models.gpt2_time import GPT2TimeModel as LM
            except Exception:
                pass

            if LM is None:
                # If your class name/path differs, change it above.
                raise ImportError("Could not import GPT2TimeModel; verify model class/import path.")

            device = torch.device("cpu")  # use "cuda" if allowed
            model = LM.load_from_checkpoint(lm_ckpt, map_location=device)
            model.eval()

            # Expect a method on the model to score sequences (adapt name if different)
            if hasattr(model, "score_sequences"):
                seqs = [seq for _, seq in cands]
                scores = model.score_sequences(seqs)     # <-- implement/confirm in your model
                rows = [
                    {"candidate_id": cid, "score": float(sc)}
                    for (cid, _), sc in zip(cands, scores)
                ]
            else:
                # If your model uses a datamodule + Trainer.predict, plug those in here.
                # from vaxseer.data.build_dm import build_predict_datamodule
                # from pytorch_lightning import Trainer
                # dm = build_predict_datamodule(cands, circ_fasta=circ_fasta, ...)
                # trainer = Trainer(logger=False, enable_checkpointing=False)
                # outputs = trainer.predict(model, dm)
                # scores = postprocess(outputs)
                # rows = [{"candidate_id": cid, "score": float(sc)} for (cid, _), sc in zip(cands, scores)]
                raise NotImplementedError("Implement your predict/postprocess path (datamodule or model scorer).")

        except Exception as e:
            print(f"[run_dominance][flu_lm][FALLBACK] {e}")
            print("[run_dominance] Falling back to placeholder scores.")
            rows = [{"candidate_id": cid, "score": float(len(seq))} for cid, seq in cands]
    else:
        rows = [{"candidate_id": cid, "score": float(len(seq))} for cid, seq in cands]

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["candidate_id", "score"])
        w.writeheader(); w.writerows(rows)

    print(f"[run_dominance] Wrote {len(rows)} rows -> {out_csv}")
    return out_csv
