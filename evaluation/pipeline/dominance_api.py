
import os
import csv
from typing import List, Tuple
from Bio import SeqIO

def _read_fasta(path: str) -> List[Tuple[str, str]]:
    """Retourne [(id, seq), ...] pour un FASTA donné."""
    recs = []
    for rec in SeqIO.parse(path, "fasta"):
        recs.append((rec.id, str(rec.seq)))
    return recs

def run_dominance(
    cand_fasta: str,
    circ_fasta: str,
    lm_ckpt: str,
    out_csv: str,
    mode: str = "placeholder",   # <--- NOUVEAU: 'placeholder' (par défaut) ou 'flu_lm'
) -> str:
    """
    API Dominance pour R/Shiny.

    - mode='placeholder' : score = longueur de séquence (permet de valider l'intégration R<->Python)
    - mode='flu_lm'      : (prochaine étape) appelle le vrai modèle GPT2TimeModel.load_from_checkpoint(...)
    """

    # --- I/O & garde-fous ---
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if not os.path.isfile(cand_fasta):
        raise FileNotFoundError(f"[run_dominance] Candidate FASTA not found: {cand_fasta}")
    if circ_fasta and not os.path.isfile(circ_fasta):
        # pas bloquant pour le placeholder, mais utile pour le modèle réel
        print(f"[run_dominance][WARN] Circulating FASTA not found: {circ_fasta}")
    if not lm_ckpt or not os.path.isfile(lm_ckpt):
        print(f"[run_dominance][WARN] LM checkpoint not found or empty path: {lm_ckpt}")

    print(f"[run_dominance] mode={mode}")
    print(f"[run_dominance] cand_fasta={cand_fasta}")
    print(f"[run_dominance] circ_fasta={circ_fasta}")
    print(f"[run_dominance] lm_ckpt={lm_ckpt}")
    print(f"[run_dominance] out_csv={out_csv}")

    # --- Lecture des séquences candidates ---
    cands = _read_fasta(cand_fasta)
    if not cands:
        raise RuntimeError(f"[run_dominance] No candidates in {cand_fasta}")

    rows = []

    if mode == "placeholder":
        # =========================
        #  PLACEHOLDER (actuel)
        # =========================
        for cid, seq in cands:
            score = float(len(seq))
            rows.append({"candidate_id": cid, "score": score})

    elif mode == "flu_lm":
        # ==========================================================
        #  ETAPE SUIVANTE : INFÉRENCE RÉELLE AVEC flu_lm (GPT2Time)
        #  -> Nous compléterons ce bloc dès que l’entrée Python
        #     de la pipeline dominance est confirmée (module/classe).
        # ==========================================================
        try:
            import torch
            # Exemple d’import probable ; sera ajusté avec TON dépôt :
            # from vaxseer.models.gpt2_time import GPT2TimeModel

            # 1) Charger le modèle Lightning depuis le .ckpt
            # model = GPT2TimeModel.load_from_checkpoint(lm_ckpt, map_location="cpu")
            # model.eval()

            # 2) Préparer le dataset de prédiction à partir des séquences candidates
            #    (l’API exacte dépend du DataModule du repo ; on l’ajoutera)
            # predict_dm = build_predict_datamodule(cands, circ_fasta, ...)

            # 3) Lancer la prédiction (Lightning Trainer)
            # from pytorch_lightning import Trainer
            # trainer = Trainer(logger=False, enable_checkpointing=False)
            # outputs = trainer.predict(model, predict_dm)

            # 4) Convertir outputs -> scores
            # rows = [{"candidate_id": cid, "score": float(score)} for (cid, _), score in zip(cands, scores)]

            raise NotImplementedError(
                "[run_dominance][flu_lm] Wiring to the real model is pending "
                "(need the exact predictor entry point from evaluation/pipeline/run.sh)."
            )

        except Exception as e:
            # On garde l'app robuste : si l'inférence réelle n'est pas prête, on retombe sur le placeholder
            print(f"[run_dominance][flu_lm][FALLBACK] {e}")
            print("[run_dominance] Falling back to placeholder scores.")
            rows = [{"candidate_id": cid, "score": float(len(seq))} for cid, seq in cands]

    else:
        raise ValueError(f"[run_dominance] Unknown mode='{mode}'. Use 'placeholder' or 'flu_lm'.")

    # --- Ecriture CSV ---
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["candidate_id", "score"])
        w.writeheader()
        w.writerows(rows)

    print(f"[run_dominance] Wrote {len(rows)} rows -> {out_csv}")
    return out_csv
