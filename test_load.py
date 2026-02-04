# test_load.py
from load_vaxseer_lm import load_gpt2_time_from_checkpoint, score_sequences_if_available

ckpt = "runs/flu_lm/2003-10_to_2018-02_2M/a_h3n2/human_minBinSize100_lenQuantile0.2/weight_loss_by_count/lightning_logs/version_0/checkpoints/epoch=99-step=101100.ckpt"
model, info = load_gpt2_time_from_checkpoint(
    ckpt_path=ckpt,
    vaxseer_root="VaxSeer",     # set to None if Shiny already inserted sys.path
    map_location="cpu",
    strict=False
)

print("Loaded ✓")
print("Used hparams:", info["used_hparams"])
print("Missing keys:", info["missing_keys"])
print("Unexpected keys:", info["unexpected_keys"])

# If the model provides score_sequences:
try:
    demo = ["MKTIIALSYIFCLVFADYKDDDDK...", "MKAILVVLLYTATNADT..."]  # two demo AA sequences
    scores = score_sequences_if_available(model, demo)
    print("Scores:", scores)
except NotImplementedError as e:
    print(str(e))
