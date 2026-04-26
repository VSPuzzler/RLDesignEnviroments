# TRIBE v2 sidecar

A FastAPI service that wraps Meta's
[`facebook/tribev2`](https://huggingface.co/facebook/tribev2) brain-encoding
model. It is **separate from the main NeuroUI Judge app** because TRIBE pins
PyTorch ≥ 2.5.1 < 2.7, which has no Python 3.14 wheel — the main app runs
on 3.14, and this sidecar runs on 3.11 inside its own conda env.

## One-time setup

```bash
# 1. Create the dedicated env (reuse if it exists).
conda create -y -n neuroui-tribe python=3.11
conda activate neuroui-tribe

# 2. Install TRIBE v2 in editable mode from the cloned repo.
git clone --depth=1 https://github.com/facebookresearch/tribev2 \
    /Users/surya/hackathons/RLDesignEnviroments/external/tribev2
pip install -e /Users/surya/hackathons/RLDesignEnviroments/external/tribev2

# 3. Install this sidecar's own deps.
pip install -r services/tribe_v2_sidecar/requirements.txt

# 4. HuggingFace login. TRIBE depends on gated meta-llama/Llama-3.2-3B.
#    Get a token at https://huggingface.co/settings/tokens (scope: read).
#    Request access to https://huggingface.co/meta-llama/Llama-3.2-3B first.
huggingface-cli login
```

## Run

```bash
conda activate neuroui-tribe
cd /Users/surya/hackathons/RLDesignEnviroments/neuro-ui-judge
uvicorn services.tribe_v2_sidecar.main:app --host 0.0.0.0 --port 7860
```

The first `POST /predict-text-with-rois` triggers a one-shot ~12 GB download
(TRIBE 0.7 GB + LLaMA-3.2-3B ~6 GB + V-JEPA2-Giant ~4 GB + W2v-BERT ~2 GB).
Subsequent calls are seconds.

## Endpoints

- `GET /health` — liveness, model load state, ROI list.
- `POST /predict-text` — raw `(n_segments, 20484)` per-segment vertex activations.
- `POST /predict-text-with-rois` — same payload **plus** the 7-channel ROI
  summary in NeuroUI Judge's schema, ready for the reward model and the
  3D cortical heatmap.

## How the main app talks to it

`services/scorer/tribe_v2_backend.py` HTTP-POSTs `/predict-text-with-rois`
on every candidate. It reads `TRIBE_V2_SERVICE_URL` from the project `.env`
(default `http://localhost:7860`). When unreachable, the backend falls back
to the deterministic mock.

## License

This service depends on `facebook/tribev2` which is released under
**CC-BY-NC-4.0** (research / non-commercial only). Read the upstream
license before any deployment.
