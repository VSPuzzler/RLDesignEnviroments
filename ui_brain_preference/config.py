"""Configuration: API keys, model IDs, paths, and ROI vertex definitions."""

import os

# ── .env autoload ────────────────────────────────────────────────────────────
# Try to load `.env` sitting next to this file. Already-exported environment
# variables win, so a shell `export` always overrides the file.
try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]

    _ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_ENV_PATH):
        load_dotenv(_ENV_PATH, override=False)
except ImportError:
    # python-dotenv not installed: fall back to whatever is already exported.
    pass

# ── API ──────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

# Model used to generate UI variants (needs strong HTML/CSS ability)
GENERATION_MODEL: str = os.getenv(
    "GENERATION_MODEL", "anthropic/claude-sonnet-4.5"
)
# Model used as the judge agent
JUDGE_MODEL: str = os.getenv(
    "JUDGE_MODEL", "anthropic/claude-sonnet-4.5"
)

# ── Tribe V2 ─────────────────────────────────────────────────────────────────
TRIBE_MODEL_ID: str = "facebook/tribev2"
# Tribe outputs predictions for fsaverage5: 20484 cortical vertices total
# (10242 left hemisphere + 10242 right hemisphere)
TRIBE_N_VERTICES: int = 20484
TRIBE_VERTICES_PER_HEMI: int = 10242

# ── Output paths ─────────────────────────────────────────────────────────────
OUTPUT_DIR: str = os.path.join(os.path.dirname(__file__), "outputs")
UI_VARIANTS_DIR: str = os.path.join(OUTPUT_DIR, "ui_variants")
SCREENSHOTS_DIR: str = os.path.join(OUTPUT_DIR, "screenshots")
COMPARISONS_FILE: str = os.path.join(OUTPUT_DIR, "comparisons.json")

# ── ROI vertex indices (fsaverage5, approximate functional parcellation) ─────
# These approximate the cortical topology of fsaverage5 using known functional
# organization (occipital→temporal→parietal→frontal ordering per hemisphere).
# In production, replace with indices from a validated atlas such as HCP MMP 1.0.

def _bilateral(lh_start: int, lh_end: int) -> list[int]:
    """Return vertex indices for both hemispheres given left-hemisphere range."""
    rh_offset = TRIBE_VERTICES_PER_HEMI
    lh = list(range(lh_start, lh_end))
    rh = list(range(rh_offset + lh_start, rh_offset + lh_end))
    return lh + rh


# Primary visual cortex (occipital pole, V1/V2)
V1_VERTICES: list[int] = _bilateral(1000, 1500)
# V4 – color / form processing (ventral occipital)
V4_VERTICES: list[int] = _bilateral(1500, 2000)
# Fusiform face area (ventral temporal cortex)
FFA_VERTICES: list[int] = _bilateral(2500, 3000)
# Reward pathway – represented on cortex via OFC / vmPFC
REWARD_VERTICES: list[int] = _bilateral(5500, 6000)
# Dorsolateral prefrontal cortex
PFC_VERTICES: list[int] = _bilateral(7000, 7500)
# Default mode network (medial PFC + PCC + angular gyrus combined)
DMN_VERTICES: list[int] = (
    _bilateral(8000, 8200)   # medial PFC
    + _bilateral(3000, 3200)  # posterior cingulate
    + _bilateral(4000, 4200)  # angular gyrus
)
# Dorsal attention network (IPS + FEF)
ATN_VERTICES: list[int] = _bilateral(6500, 7000)
