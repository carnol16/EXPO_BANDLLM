import os
import wave
import random
import datetime

BACKING_VOCAL_CHANCE = 0.70
MAX_RETRIES_PER_LINE = 3


def _load_kokoro_pipeline(voice_id):
    import kokoro
    try:
        return kokoro.KPipeline(lang_code="a", voice=voice_id, repo_id="hexgrad/Kokoro-82M", device="cpu")
    except TypeError:
        pipe = kokoro.KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device="cpu")
        pipe.voice = voice_id
        return pipe


def _synthesize_line(pipeline, text):
    import numpy as np
    audio_chunks = []
    sample_rate = 24000
    call_kwargs = {"speed": 1.0}
    voice_id = getattr(pipeline, "voice", None)
    if voice_id is not None:
        call_kwargs["voice"] = voice_id
    for _, _, audio in pipeline(text, **call_kwargs):
        audio_chunks.append(audio)
    if not audio_chunks:
        raise RuntimeError("Pipeline returned no audio chunks")
    samples = np.concatenate(audio_chunks)
    pcm = (samples * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
    return pcm, sample_rate


def _write_wav(pcm_bytes, sample_rate, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def _concat_wavs(wav_paths, output_path):
    if not wav_paths:
        return
    params = None
    all_frames = []
    for path in wav_paths:
        try:
            with wave.open(path, "rb") as wf:
                p = wf.getparams()
                if params is None:
                    params = p
                elif (p.nchannels, p.sampwidth, p.framerate) != (params.nchannels, params.sampwidth, params.framerate):
                    print(f"[VocalSynth] Warning: {path} has mismatched params, skipping")
                    continue
                all_frames.append(wf.readframes(wf.getnframes()))
        except Exception as e:
            print(f"[VocalSynth] Warning: could not read {path}: {e}")
    if not all_frames or params is None:
        return
    with wave.open(output_path, "wb") as wf:
        wf.setparams(params)
        for frames in all_frames:
            wf.writeframes(frames)
    for path in wav_paths:
        try:
            os.remove(path)
        except OSError:
            pass


def synthesize_line_with_retries(pipeline, text, line_index, author):
    for attempt in range(1, MAX_RETRIES_PER_LINE + 1):
        print(f"[VocalSynth] Synthesizing {author} line {line_index} (attempt {attempt}/{MAX_RETRIES_PER_LINE}): '{text[:40]}...'")
        try:
            return _synthesize_line(pipeline, text)
        except Exception as e:
            print(f"[VocalSynth] Attempt {attempt} failed: {e}")
    print(f"[VocalSynth] FAILED: {author} line {line_index} after {MAX_RETRIES_PER_LINE} attempts — skipping")
    return None


def synthesize_vocals(song_doc, voice_config, day_name, output_dir):
    vocals_dir = os.path.join(output_dir, f"vocals_{day_name}")
    os.makedirs(vocals_dir, exist_ok=True)
    result = {"lead": None, "backing": None}

    # Load pipelines for each character
    pipelines = {}
    for character, cfg in voice_config.items():
        try:
            pipelines[character] = _load_kokoro_pipeline(cfg["voice_id"])
        except Exception as e:
            print(f"[VocalSynth] Warning: could not load pipeline for {character}: {e}")

    # Lead track — singer lines
    singer_lines = [l for l in song_doc.lyrics if l.get("author") == "singer"]
    if not singer_lines:
        print("[VocalSynth] Warning: no singer lines found, skipping lead synthesis")
    else:
        segments = []
        singer_pipeline = pipelines.get("singer")
        if singer_pipeline:
            for i, lyric in enumerate(singer_lines):
                res = synthesize_line_with_retries(singer_pipeline, lyric["text"], i, "singer")
                if res is not None:
                    seg_path = os.path.join(vocals_dir, f"lead_{i:03d}.wav")
                    _write_wav(res[0], res[1], seg_path)
                    segments.append(seg_path)

        if not segments:
            print("[VocalSynth] Singer lead track empty — running full retry pass with default parameters")
            try:
                fresh_pipeline = _load_kokoro_pipeline(voice_config["singer"]["voice_id"])
                for i, lyric in enumerate(singer_lines):
                    try:
                        res = _synthesize_line(fresh_pipeline, lyric["text"])
                        seg_path = os.path.join(vocals_dir, f"lead_retry_{i:03d}.wav")
                        _write_wav(res[0], res[1], seg_path)
                        segments.append(seg_path)
                    except Exception as e:
                        print(f"[VocalSynth] Retry failed for line {i}: {e}")
            except Exception as e:
                print(f"[VocalSynth] Full retry pass failed: {e}")

        if segments:
            lead_path = os.path.join(vocals_dir, "vocals_lead.wav")
            _concat_wavs(segments, lead_path)
            result["lead"] = lead_path
        else:
            print("[VocalSynth] Full retry pass failed — falling back to prompt-only generation")

    # Backing track — non-singer lines
    backing_lines = [l for l in song_doc.lyrics if l.get("author") != "singer"]
    if not backing_lines:
        print("[VocalSynth] No backing lines found, skipping backing synthesis")
    else:
        segments = []
        for i, lyric in enumerate(backing_lines):
            author = lyric.get("author", "unknown")
            if random.random() >= BACKING_VOCAL_CHANCE:
                print(f"[VocalSynth] {author} line {i} skipped by chance roll")
                continue
            pipeline = pipelines.get(author)
            if pipeline is None:
                print(f"[VocalSynth] No pipeline for {author}, skipping line {i}")
                continue
            res = synthesize_line_with_retries(pipeline, lyric["text"], i, author)
            if res is not None:
                seg_path = os.path.join(vocals_dir, f"backing_{i:03d}.wav")
                _write_wav(res[0], res[1], seg_path)
                segments.append(seg_path)

        if segments:
            backing_path = os.path.join(vocals_dir, "vocals_backing.wav")
            _concat_wavs(segments, backing_path)
            result["backing"] = backing_path

    print(f"[VocalSynth] Complete:\n  Lead:    {result['lead'] or 'FAILED'}\n  Backing: {result['backing'] or 'none'}")
    return result
