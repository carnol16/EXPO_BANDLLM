"""Vanilla ACE-Step 1.5 test — no custom split, mirrors the CLI pipeline exactly."""
import os
import sys
import datetime

_ACE_STEP_15_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ACE-Step-1.5")
if _ACE_STEP_15_PATH not in sys.path:
    sys.path.insert(0, _ACE_STEP_15_PATH)

os.environ.setdefault("ACESTEP_GENERATION_TIMEOUT", "3600")

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

CAPTION = (
    "A relentless punk-industrial track at 120 BPM. D-beat drums hammering without let-up, "
    "heavily distorted downtuned guitar playing fast power chords and grinding riffs, "
    "overdriven bass locked to the kick, raw harsh screamed vocals with no reverb polish. "
    "Lo-fi recording aesthetic — natural room noise, no studio sheen, everything sounds "
    "live and brutal. Sonic texture: mechanical, corroded, abrasive. "
    "Influences: early Discharge, Killing Joke, Godflesh. "
    "Feedback swells between sections, amp hum in the gaps."
)

LYRICS = """[Intro]

[Verse 1]
Fractured cadence, the last heartbeat of a dying empire
Machines replace the workers, rust replaces the dream

[Chorus 1]
SYSTEMS DEVOUR THEIR OWN OPERATORS
COLD ENTROPY UNFOLDING LIKE RUST ON STEEL

[Verse 2]
Fractured cadence, circuits burned and broken
Silent floors where voices used to echo

[Chorus 2]
There is no signal left to amplify
Only the drone of the machine remains

[Bridge]
SYSTEMS DEVOUR THEIR OWN OPERATORS
COLD ENTROPY UNFOLDING LIKE RUST ON STEEL

[Chorus 3]
We were the last transmission
SYSTEMS DEVOUR THEIR OWN OPERATORS
COLD ENTROPY UNFOLDING LIKE RUST ON STEEL

[Outro]"""

OUTPUT_DIR = os.path.join("output", f"vanilla_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

checkpoint_dir = os.path.join(_ACE_STEP_15_PATH, "checkpoints")

# --- Init DiT handler ---
print("[Vanilla] Initializing DiT handler...")
dit_handler = AceStepHandler()
status, ok = dit_handler.initialize_service(
    project_root=_ACE_STEP_15_PATH,
    config_path="acestep-v15-sft",
    device="auto",
    offload_to_cpu=True,
)
if not ok:
    print(f"[Vanilla] DiT init failed: {status}")
    sys.exit(1)
print("[Vanilla] DiT ready")

# --- Init LLM handler ---
print("[Vanilla] Initializing LLM handler...")
llm_handler = LLMHandler(persistent_storage_path=_ACE_STEP_15_PATH)
available_lm = llm_handler.get_available_5hz_lm_models()
if not available_lm:
    print("[Vanilla] No 5Hz LM models found — aborting")
    sys.exit(1)

lm_status, lm_ok = llm_handler.initialize(
    checkpoint_dir=checkpoint_dir,
    lm_model_path=available_lm[0],
    backend="pt",
    device="auto",
    offload_to_cpu=False,
)
if not lm_ok:
    print(f"[Vanilla] LLM init failed: {lm_status}")
    sys.exit(1)
print("[Vanilla] LLM ready")

# --- Build params — thinking=True, let the pipeline handle CoT internally ---
params = GenerationParams(
    caption=CAPTION,
    lyrics=LYRICS,
    bpm=120,
    keyscale="D minor",
    timesignature="",
    duration=150.0,
    thinking=True,        # LM runs CoT internally, not pre-split
    audio_codes="",
    use_cot_metas=True,
    use_cot_caption=True,
    use_cot_language=True,
    inference_steps=60,
    guidance_scale=7.0,
    lm_negative_prompt="distorted speech, off-key singing, low quality, noise, silence",
)

config = GenerationConfig(
    batch_size=1,
    audio_format="wav",
    use_random_seed=True,
)

print("[Vanilla] Starting generation (thinking=True, CoT+DiT together)...")
try:
    result = generate_music(dit_handler, llm_handler, params, config, save_dir=OUTPUT_DIR)
except Exception as e:
    print(f"[Vanilla] Generation failed: {e}")
    sys.exit(1)

if not result.success or not result.audios:
    print(f"[Vanilla] Failed: {result.error or 'no audios'}")
    sys.exit(1)

audio_path = result.audios[0].get("path", "")
print(f"\n[Vanilla] RESULT: {audio_path}")
