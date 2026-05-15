"""Quick smoke test for ace_step_bridge.generate()"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from document_creator.song_document import SongDocument
import ace_step_bridge

doc = SongDocument()
doc.set_field("title", "Test Track")
doc.set_field("tempo_bpm", 120)
doc.set_field("key", "D minor")
doc.set_field("mood", "ferocious, dark")
doc.set_field("instruments", ["distorted guitar", "d-beat drums", "bass", "screamed vocals"])
doc.set_field("vibe_notes", "raw, industrial, corroded")
doc.set_field("acestep_caption",
    "A relentless punk-industrial track at 120 BPM. D-beat drums hammering without let-up, "
    "heavily distorted downtuned guitar playing fast power chords and grinding riffs, "
    "overdriven bass locked to the kick, raw harsh screamed vocals with no reverb polish. "
    "Lo-fi recording aesthetic — natural room noise, everything sounds "
    "live and brutal. Sonic texture: mechanical, corroded, abrasive. "
    "Structure moves through verse and chorus without dynamic softening — stays loud and aggressive throughout. "
    "Influences: early Discharge, Killing Joke, Godflesh. "
    "Feedback swells between sections, amp hum in the gaps."
)
doc.set_field("structure", ["intro", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "outro"])
doc.set_field("lyrics", [
    {"author": "singer", "text": "Fractured cadence, the last heartbeat of a dying empire"},
    {"author": "singer", "text": "Machines replace the workers, rust replaces the dream"},
    {"author": "singer", "text": "SYSTEMS DEVOUR THEIR OWN OPERATORS"},
    {"author": "singer", "text": "COLD ENTROPY UNFOLDING LIKE RUST ON STEEL"},
    {"author": "singer", "text": "Fractured cadence, circuits burned and broken"},
    {"author": "singer", "text": "Silent floors where voices used to echo"},
    {"author": "singer", "text": "There is no signal left to amplify"},
    {"author": "singer", "text": "Only the drone of the machine remains"},
    {"author": "singer", "text": "SYSTEMS DEVOUR THEIR OWN OPERATORS"},
    {"author": "singer", "text": "COLD ENTROPY UNFOLDING LIKE RUST ON STEEL"},
    {"author": "singer", "text": "We were the last transmission"},
    {"author": "singer", "text": "SYSTEMS DEVOUR THEIR OWN OPERATORS"},
    {"author": "singer", "text": "COLD ENTROPY UNFOLDING LIKE RUST ON STEEL"},
])

estimated = ace_step_bridge._estimate_duration(doc)
print(f"Estimated duration: {estimated:.1f}s ({estimated/60:.2f} min)")

result = ace_step_bridge.generate(doc, "test", None, None, None, fast_mode=True)
print(f"\nRESULT: {result}")
