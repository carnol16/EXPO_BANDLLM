import argparse
import json
import os
import subprocess
import sys
import tempfile
import wave


CHARACTERS = ["singer", "guitarist", "bassist", "drummer"]

PROMPT_FILES = {c: f"prompts/{c}.txt" for c in CHARACTERS}


def _get_kokoro_voices():
    import kokoro
    try:
        voices = kokoro.list_voices()
        return sorted(voices)
    except AttributeError:
        pass
    try:
        pipe = kokoro.KPipeline(lang_code="a")
        voices = list(pipe.voices) if hasattr(pipe, "voices") else []
        if voices:
            return sorted(voices)
    except Exception:
        pass
    # kokoro 0.9+ lazily loads voices — KPipeline.voices starts empty.
    # Query the HF Hub for the actual list of voice files.
    try:
        from huggingface_hub import list_repo_files
        voices = [
            f.split("/")[-1].replace(".pt", "")
            for f in list_repo_files("hexgrad/Kokoro-82M")
            if f.startswith("voices/") and f.endswith(".pt")
        ]
        return sorted(voices)
    except Exception:
        pass
    return []


def _play_wav(path):
    subprocess.run(["aplay", path], check=False)


def _sample_voice(voice_id):
    import kokoro
    import numpy as np

    phrase = "Burning city, no way out. Tear it all down."
    try:
        pipe = kokoro.KPipeline(lang_code="a", voice=voice_id)
        audio_chunks = []
        sample_rate = 24000
        for _, _, audio in pipe(phrase, speed=1.0):
            audio_chunks.append(audio)
        if not audio_chunks:
            print("  (no audio generated)")
            return
        samples = np.concatenate(audio_chunks)
        pcm = (samples * 32767).clip(-32768, 32767).astype(np.int16).tobytes()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)

        _play_wav(tmp_path)
        os.remove(tmp_path)
    except Exception as e:
        print(f"  (sample playback failed: {e})")


def assign_voices():
    import kokoro

    print("Loading Kokoro voices...")
    voices = _get_kokoro_voices()
    if not voices:
        print("ERROR: Could not load Kokoro voice list. Is kokoro installed?")
        sys.exit(1)

    print(f"Found {len(voices)} voices.\n")

    config = {}

    for character in CHARACTERS:
        while True:
            print(f"\n=== Assigning voice for: {character.upper()} ===")
            for i, v in enumerate(voices):
                print(f"  {i}: {v}")

            raw = input(f"\nEnter voice ID for {character} (or number from list): ").strip()
            if raw.isdigit():
                idx = int(raw)
                if 0 <= idx < len(voices):
                    voice_id = voices[idx]
                else:
                    print("Number out of range, try again.")
                    continue
            else:
                voice_id = raw

            if voice_id not in voices:
                print(f"'{voice_id}' not in voice list, try again.")
                continue

            print(f"\nPlaying sample for '{voice_id}'...")
            _sample_voice(voice_id)

            accept = input("Accept this voice? (y/n): ").strip().lower()
            if accept != "y":
                continue

            while True:
                style = input(
                    f"Enter vocal style description for ACE-Step "
                    f"(e.g. 'intense female vocals, raw, urgent'): "
                ).strip()
                if style:
                    break
                print("Description cannot be empty.")

            config[character] = {"voice_id": voice_id, "vocal_style": style}
            break

    with open("voice_config.json", "w") as f:
        f.write(json.dumps(config, indent=2))
    print("\nvoice_config.json written successfully.")


def preflight_check(model_config_path):
    failures = []

    # Check 1 — prompt files
    for character, path in PROMPT_FILES.items():
        if os.path.isfile(path):
            print(f"[PASS] {path}")
        else:
            print(f"[FAIL] {path} — file not found")
            failures.append(path)

    # Check 2 — voice_config.json
    if not os.path.isfile("voice_config.json"):
        print("[FAIL] voice_config.json — file not found")
        failures.append("voice_config.json")
    else:
        try:
            with open("voice_config.json") as f:
                vc = json.load(f)
            missing = [c for c in CHARACTERS if c not in vc]
            if missing:
                print(f"[FAIL] voice_config.json — missing keys: {missing}")
                failures.append("voice_config.json")
            else:
                bad = [
                    c for c in CHARACTERS
                    if not vc[c].get("voice_id") or not vc[c].get("vocal_style")
                ]
                if bad:
                    print(f"[FAIL] voice_config.json — missing voice_id or vocal_style for: {bad}")
                    failures.append("voice_config.json")
                else:
                    print("[PASS] voice_config.json")
        except Exception as e:
            print(f"[FAIL] voice_config.json — parse error: {e}")
            failures.append("voice_config.json")

    # Check 3 — GGUF model files
    if not os.path.isfile(model_config_path):
        print(f"[FAIL] {model_config_path} — file not found")
        failures.append(model_config_path)
    else:
        try:
            with open(model_config_path) as f:
                mc = json.load(f)
            for key, path in mc.items():
                if not path:
                    print(f"[SKIP] {key} — no path configured")
                    continue
                if os.path.isfile(path) and os.path.getsize(path) > 0:
                    print(f"[PASS] {key}: {path}")
                else:
                    print(f"[FAIL] {key}: {path} — not found or empty")
                    failures.append(key)
        except Exception as e:
            print(f"[FAIL] {model_config_path} — parse error: {e}")
            failures.append(model_config_path)

    # Check 4 — ACE-Step
    acestep_ok = False
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:8001/docs", timeout=3)
        print("[PASS] ACE-Step V1.5 (REST API server running at localhost:8001)")
        acestep_ok = True
    except Exception:
        pass
    if not acestep_ok:
        try:
            subprocess.run(["acestep-api", "--help"], capture_output=True, timeout=5)
            print("[PASS] ACE-Step V1.5 (REST API — recommended)")
            acestep_ok = True
        except FileNotFoundError:
            pass
    if not acestep_ok:
        try:
            import acestep  # noqa: F401
            print("[PASS] ACE-Step (Python import)")
            acestep_ok = True
        except ImportError:
            pass
    if not acestep_ok:
        try:
            import ace_step  # noqa: F401
            print("[PASS] ACE-Step (Python import)")
            acestep_ok = True
        except ImportError:
            pass
    if not acestep_ok:
        try:
            subprocess.run(["acestep", "--help"], capture_output=True, timeout=5)
            print("[PASS] ACE-Step (CLI)")
            acestep_ok = True
        except FileNotFoundError:
            pass
    if not acestep_ok:
        print("[FAIL] ACE-Step — not found. Install ACE-Step V1.5: https://github.com/ace-step/ACE-Step-1.5")
        failures.append("acestep")

    # Check 5 — Kokoro
    try:
        import kokoro  # noqa: F401
        print("[PASS] Kokoro TTS")
    except ImportError:
        print("[FAIL] Kokoro TTS — run: pip install kokoro")
        failures.append("kokoro")

    # Check 6 — output/ directory
    os.makedirs("output", exist_ok=True)
    print("[PASS] output/ directory")

    # Summary
    print()
    if not failures:
        print("ALL CHECKS PASSED — ready to launch.")
        print()
        print("Launch command:")
        print("  python band_coordinator.py --model-config model_config.json --osc-ip <ip> --port 8765")
        print("  (replace <ip> with your OSC target IP address)")
    else:
        print("SOME CHECKS FAILED — fix issues above before launching.")


def main():
    parser = argparse.ArgumentParser(description="Band installation setup and preflight check.")
    parser.add_argument("--assign-voices", action="store_true", help="Run interactive voice assignment")
    parser.add_argument("--check", action="store_true", help="Run preflight checks only")
    parser.add_argument("--model-config", type=str, default="model_config.json", help="Path to model config JSON")
    args = parser.parse_args()

    if not args.assign_voices and not args.check:
        parser.print_help()
        sys.exit(0)

    if args.assign_voices:
        assign_voices()

    preflight_check(args.model_config)


if __name__ == "__main__":
    main()
