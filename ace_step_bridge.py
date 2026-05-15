import datetime
import os
import sys

# Extend the generation wall-clock timeout — default 600s is too tight when
# bots hold VRAM and each diffusion step competes for memory.
os.environ.setdefault("ACESTEP_GENERATION_TIMEOUT", "3600")

OUTPUT_DIR = "output"
_MIN_DURATION = 180         # 3 minutes minimum
_MAX_DURATION = 300         # 5 minutes maximum
_INFERENCE_STEPS = 8        # turbo dmd_gan: hard max is 8 steps (higher values are clamped)
_INFERENCE_STEPS_FAST = 8  # same cap

# DiT model to use. Swap to test variants (all turbo = 8 steps max, no CFG):
#   "acestep-v15-turbo"            — default, best balance of creativity + semantics
#   "acestep-v15-turbo-shift3"     — clearer/richer timbre, but minimal orchestration
#   "acestep-v15-turbo-continuous" — experimental, continuous shift 1-5, most flexible
_DIT_MODEL = "acestep-v15-turbo"

# Prepend local ACE-Step 1.5 to sys.path so it shadows the old pip-installed 0.2.x
_ACE_STEP_15_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ACE-Step-1.5")
if _ACE_STEP_15_PATH not in sys.path:
    sys.path.insert(0, _ACE_STEP_15_PATH)

_dit_handler = None
_NUM_VARIANTS = 3   # audio variants per track — user picks best for listening party


def _detect_torch_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' based on what PyTorch can see at runtime."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _flush_torch_cache(device: str):
    """Release the allocator cache for the active device."""
    try:
        import torch
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.empty_cache()
    except Exception:
        pass


def _vram_report(label):
    try:
        import torch
        device = _detect_torch_device()
        if device == "cuda":
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free = total - reserved
            print(f"[VRAM] {label}: {alloc:.2f}GB alloc / {reserved:.2f}GB reserved / {free:.2f}GB free / {total:.2f}GB total")
        elif device == "mps":
            alloc = torch.mps.current_allocated_memory() / 1024**3
            print(f"[VRAM] {label}: MPS unified memory — {alloc:.2f}GB allocated by PyTorch")
        else:
            print(f"[VRAM] {label}: CPU-only")
    except Exception as e:
        print(f"[VRAM] {label}: error — {e}")


def _release_dit():
    """Null out all DiT model components and flush the CUDA allocator.

    Call this before running CoT (so LM gets full VRAM) and after each
    track generation completes (so the NEXT call's CoT also gets full VRAM).
    Re-initialization adds ~2-3 min per track — acceptable for once-per-day use.
    """
    global _dit_handler
    import gc 
    import torch
    if _dit_handler is None:
        return
    print("[ACEStep-1.5] Releasing DiT VRAM...")
    for attr in ("model", "vae", "text_encoder", "text_tokenizer",
                 "silence_latent", "reward_model"):
        try:
            if getattr(_dit_handler, attr, None) is not None:
                setattr(_dit_handler, attr, None)
        except Exception:
            pass
    _dit_handler = None
    gc.collect()
    _flush_torch_cache(_detect_torch_device())
    print("[ACEStep-1.5] DiT VRAM released")


def _get_dit_handler():
    """Return the cached DiT handler, initializing it if needed.

    Call this AFTER _run_cot() has completed and freed its VRAM so DiT has
    the full 10 GB budget.  DiT is kept alive as a singleton across calls
    because re-initialization takes several minutes.
    """
    global _dit_handler
    if _dit_handler is not None:
        return _dit_handler

    from acestep.handler import AceStepHandler

    device = _detect_torch_device()
    # CUDA: offload DiT back to CPU after diffusion so VAE decode gets the full VRAM budget.
    # MPS / CPU: unified memory — CPU and GPU share the same pool, so offloading is pointless.
    cuda_offload = device == "cuda"
    print(f"[ACEStep-1.5] Initializing DiT handler (device={device}, offload={cuda_offload})...")
    _dit_handler = AceStepHandler()
    status, ok = _dit_handler.initialize_service(
        project_root=_ACE_STEP_15_PATH,
        config_path=_DIT_MODEL,
        device="auto",
        offload_to_cpu=cuda_offload,
        offload_dit_to_cpu=cuda_offload,
    )
    if not ok:
        print(f"[ACEStep-1.5] DiT init failed: {status}")
        _dit_handler = None
        return None
    print("[ACEStep-1.5] DiT ready")
    return _dit_handler


def _run_cot(caption, lyrics, duration, bpm, keyscale, lm_negative_prompt):
    """Run the 5Hz LM CoT metadata phase on GPU alone, then free all VRAM before DiT loads.

    Uses infer_type="dit" (metadata only — no audio code generation) because DiT
    runs text2music from scratch.  Returns metadata_dict or None on failure.

    VRAM sequence:
      LLM (~3.5 GB) loads → CoT runs (metadata: BPM, key, duration) →
      LLM.unload() + empty_cache → DiT (~4.5 GB) loads in _get_dit_handler().
    """
    import gc
    from acestep.llm_inference import LLMHandler

    checkpoint_dir = os.path.join(_ACE_STEP_15_PATH, "checkpoints")

    llm_handler = LLMHandler(persistent_storage_path=_ACE_STEP_15_PATH)
    available_lm = llm_handler.get_available_5hz_lm_models()
    if not available_lm:
        print("[ACEStep-1.5] CoT: No 5Hz LM models found — skipping CoT")
        return None

    lm_model = available_lm[-1]  # prefer largest available (sorted: 0.6B < 1.7B < 4B)
    print(f"[ACEStep-1.5] CoT: Loading LLM onto GPU ({os.path.basename(lm_model)})...")
    lm_status, lm_ok = llm_handler.initialize(
        checkpoint_dir=checkpoint_dir,
        lm_model_path=lm_model,
        backend="pt",
        device="auto",
        offload_to_cpu=False,
    )
    if not lm_ok:
        print(f"[ACEStep-1.5] CoT: LLM init failed ({lm_status}) — skipping CoT")
        return None

    user_metadata = {}
    if bpm:
        user_metadata["bpm"] = int(bpm)
    if keyscale and keyscale.strip():
        user_metadata["keyscale"] = keyscale.strip()
    if duration and duration > 0:
        user_metadata["duration"] = int(duration)

    print("[ACEStep-1.5] CoT: Running metadata inference (BPM / key / duration)...")
    metadata = None
    try:
        result = llm_handler.generate_with_stop_condition(
            caption=caption,
            lyrics=lyrics,
            infer_type="dit",       # metadata only — no audio code generation
            temperature=0.67,
            cfg_scale=2.0,
            negative_prompt=lm_negative_prompt,
            top_k=None,
            top_p=0.9,
            target_duration=duration,
            user_metadata=user_metadata if user_metadata else None,
            use_cot_caption=True,
            use_cot_language=True,
            use_cot_metas=True,
            use_constrained_decoding=False,
            batch_size=16,
            seeds=None,
            progress=None,
        )
        if not result.get("success", False):
            print(f"[ACEStep-1.5] CoT: LLM generation failed: {result.get('error', 'unknown')}")
        else:
            metadata = result.get("metadata", {})
            if isinstance(metadata, list):
                metadata = metadata[0] if metadata else {}
            print(f"[ACEStep-1.5] CoT: Complete — metadata: {metadata}")
    except Exception as e:
        print(f"[ACEStep-1.5] CoT: Exception during generation: {e}")
    finally:
        print("[ACEStep-1.5] CoT: Unloading LLM to free VRAM for DiT...")
        try:
            llm_handler.unload()
        except Exception:
            pass
        gc.collect()
        _flush_torch_cache(_detect_torch_device())
        print("[ACEStep-1.5] CoT: LLM unloaded")

    return metadata


def _estimate_duration(song_doc):
    bpm = song_doc.tempo_bpm or 140
    secs_per_bar = 60.0 / bpm * 4

    _LONG = {"verse", "chorus", "bridge", "pre-chorus", "hook", "refrain"}
    _SHORT = {"intro", "outro", "instrumental", "solo", "interlude", "breakdown"}

    structure = song_doc.structure or []
    if structure:
        bars = sum(32 if s in _LONG else 16 for s in structure)
    else:
        bars = 10 * 24

    return min(_MAX_DURATION, max(_MIN_DURATION, bars * secs_per_bar))


def _build_fallback_caption(song_doc):
    """Build a multi-dimension caption when no distilled caption exists."""
    mood = song_doc.mood or "aggressive, confrontational"
    vibe = song_doc.vibe_notes

    parts = [
        # Style + emotion
        f"Deathcore, Progressive metal, and grunge. {mood}. Relentless and abrasive.",
        # Instruments + timbre (repetition reinforcement on distortion/grit)
        "Heavily distorted electric guitar with high-gain tube saturation, palm-muted chugs, crunchy power chords.",
        "Overdriven gritty bass, punchy low-end. Live d-beat drum kit, loud and driving.",
        # Vocal
        "Male screamed vocals, raw, harsh, confrontational.",
        # Production + era
        "Analog recording, natural noise floor, live room sound. Late 80s hardcore industrial energy.",
    ]

    if vibe:
        parts.append(f"Texture: {vibe}.")

    caption = " ".join(parts)
    return caption[:600]


def generate(song_doc, day_name, osc_sender, resource_manager, vocal_paths, fast_mode=False):
    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    # Free DiT VRAM before CoT — on subsequent days the singleton is still loaded
    # and would leave only ~352 MB free, making CoT generation nearly impossible.
    _vram_report("start of generate()")
    _release_dit()
    _vram_report("after DiT release")

    if not song_doc.is_complete():
        print("[ACEStep-1.5] WARNING: song_document incomplete — generating with available fields")
        for field in ("title", "tempo_bpm", "key", "mood"):
            if getattr(song_doc, field) is None:
                print(f"[ACEStep-1.5]   missing: {field}")

    # Caption: use LLM-distilled caption if available, else build fallback.
    # Always prepend a genre anchor so the DiT model cannot drift to calm/ambient
    # regardless of what the distillation LLM wrote.
    _GENRE_PREFIX = (
    "90's grunge, 2010's deathcore. in the vein of Nirvana, Alice and Chains, early TOOL (Undertow era), chelsea grin, and Born of Osiris "
    "Heavily distorted electric guitar: high-gain tube saturation, buzzsaw tone, palm-muted power chord chugs, abrasive. "
    "Overdriven gritty bass, punchy low-end. "
    "Live d-beat drum kit, along with double kicks, loud and driving, crushing half-time breakdown. "
    "Male screamed vocals, raw and confrontational. "
    "Analog recording, live room sound, natural noise floor. "
    )
    _base_caption = song_doc.acestep_caption or _build_fallback_caption(song_doc)
    caption = (_GENRE_PREFIX + _base_caption)[:1024]
    lyrics = song_doc.to_acestep_lyrics()

    print(f"[ACEStep-1.5] Caption:\n{caption}")
    print(f"[ACEStep-1.5] Lyrics:\n{lyrics}")

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    duration = _estimate_duration(song_doc)
    inference_steps = _INFERENCE_STEPS

    lm_negative_prompt = "clean guitar, synth, synthesizer, keyboard, piano, strings, pads, atmospheric, silence"

    # CoT disabled — all metadata comes directly from song_doc.
    # CoT took 265s and generated wrong BPM/key/duration that were overridden anyway.
    # Re-enable by uncommenting the block below and removing the three final_* lines.
    #
    # _vram_report("before CoT / LLM load")
    # cot_metadata = _run_cot(
    #     caption=caption,
    #     lyrics=lyrics,
    #     duration=duration,
    #     bpm=song_doc.tempo_bpm,
    #     keyscale=song_doc.key or "",
    #     lm_negative_prompt=lm_negative_prompt,
    # )
    # _vram_report("after CoT / LLM unload")
    # if cot_metadata:
    #     if not final_bpm and cot_metadata.get("bpm"):
    #         final_bpm = cot_metadata["bpm"]
    #     if not final_keyscale and cot_metadata.get("keyscale"):
    #         final_keyscale = cot_metadata["keyscale"]
    #     cot_dur = cot_metadata.get("duration", 0)
    #     if cot_dur and float(cot_dur) > 0:
    #         final_duration = min(_MAX_DURATION, max(_MIN_DURATION, float(cot_dur)))

    final_bpm = song_doc.tempo_bpm
    final_keyscale = song_doc.key or ""
    final_duration = duration

    _vram_report("before DiT init")
    dit_handler = _get_dit_handler()
    _vram_report("after DiT init")
    if dit_handler is None:
        print("[ACEStep-1.5] Handler init failed — skipping generation")
        return []

    print(f"[ACEStep-1.5] Duration: {final_duration:.1f}s ({final_duration/60:.2f} min), "
          f"steps: {inference_steps}, variants: {_NUM_VARIANTS}, task: text2music")

    params = GenerationParams(
        caption=caption,
        lyrics=lyrics,
        bpm=final_bpm,
        keyscale=final_keyscale,
        timesignature="4/4",        # d-beat/punk is always 4/4; explicit anchor helps the model
        duration=final_duration,
        vocal_language="en",
        thinking=False,
        audio_codes="",
        use_cot_metas=False,
        use_cot_caption=False,
        use_cot_language=False,
        inference_steps=inference_steps,
        guidance_scale=7.0,         # CFG only works on Base/SFT — ignored on turbo; leave at default
        lm_negative_prompt=lm_negative_prompt,  # only active if LM/CoT re-enabled
    )

    config = GenerationConfig(
        batch_size=1,
        audio_format="wav",
        use_random_seed=True,
    )

    # Generate _NUM_VARIANTS with different random seeds. DiT stays loaded between
    # calls so we only pay the init/release cost once per track session.
    variant_paths = []
    for v_idx in range(_NUM_VARIANTS):
        variant_dir = os.path.join(OUTPUT_DIR, f"track_{day_name}_{timestamp}_v{v_idx + 1}")
        os.makedirs(variant_dir, exist_ok=True)
        print(f"[ACEStep-1.5] Variant {v_idx + 1}/{_NUM_VARIANTS}...")
        try:
            result = generate_music(dit_handler, None, params, config, save_dir=variant_dir)
        except Exception as e:
            print(f"[ACEStep-1.5] Variant {v_idx + 1} failed: {e}")
            continue
        if not result.success or not result.audios:
            print(f"[ACEStep-1.5] Variant {v_idx + 1} failed: {result.error or 'no audios returned'}")
            continue
        audio_path = result.audios[0].get("path", "")
        if not audio_path or not os.path.isfile(audio_path):
            print(f"[ACEStep-1.5] Variant {v_idx + 1} audio file missing: {audio_path!r}")
            continue
        variant_paths.append(audio_path)
        print(f"[ACEStep-1.5] Variant {v_idx + 1} complete: {audio_path}")
        if osc_sender is not None:
            osc_sender.send_message("/band/track_variant", audio_path)

    # Release DiT once after all variants so the next track's CoT gets full VRAM.
    _release_dit()

    print(f"[ACEStep-1.5] {len(variant_paths)}/{_NUM_VARIANTS} variants generated for {day_name}")
    return variant_paths
