"""Quick test: load a song state JSON and run ace_step_bridge.generate()

Usage:
    python run_song_state_test.py                          # defaults to song_state.json
    python run_song_state_test.py song_states/monday_1.json
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from document_creator.song_document import SongDocument
import ace_step_bridge

parser = argparse.ArgumentParser()
parser.add_argument("--path", default="song_state.json", help="Path to song state JSON (default: song_state.json)")
args = parser.parse_args()

print(f"Loading: {args.path}")
doc = SongDocument.load(args.path)

print("=== Song State ===")
print(f"  Title:    {doc.title}")
print(f"  Genre:    {doc.genre}")
print(f"  BPM:      {doc.tempo_bpm}")
print(f"  Key:      {doc.key}")
print(f"  Mood:     {doc.mood}")
print(f"  Structure:{doc.structure}")
print(f"  Lyrics:   {len(doc.lyrics or [])} line(s)")
print(f"  Caption:  {'(distilled)' if doc.acestep_caption else '(fallback — no distilled caption saved)'}")
print()

estimated = ace_step_bridge._estimate_duration(doc)
print(f"Estimated duration: {estimated:.1f}s ({estimated/60:.2f} min)")
print()

_GENRE_PREFIX = (
    "90's grunge, progressive metal, 2010's deathcore. in the vein of Nirvana, Alice and Chains, early TOOL (Undertow era), chelsea grin, and Born of Osiris "
    "Heavily distorted electric guitar: high-gain tube saturation, buzzsaw tone, palm-muted power chord chugs, abrasive. "
    "Overdriven gritty bass, punchy low-end. "
    "Live d-beat drum kit, along with double kicks, loud and driving, crushing half-time breakdown. "
    "Male screamed vocals, raw and confrontational. "
    "Analog recording, live room sound, natural noise floor. "
)
_base = doc.acestep_caption or ace_step_bridge._build_fallback_caption(doc)
full_caption = (_GENRE_PREFIX + _base)[:1024]

print("=== Caption that will be sent to ACE-Step ===")
print(full_caption)
print()

lyrics_out = doc.to_acestep_lyrics()
print("=== Lyrics that will be sent to ACE-Step ===")
print(lyrics_out)
print()

result = ace_step_bridge.generate(doc, "song_state_test", None, None, None)
print(f"\nRESULT: {result}")
