EXPO_BANDLLM — Project Summary
What This Is
A generative art installation where 4 local LLM bots roleplay as a punk-industrial band (Rex/Singer, Volt/Guitarist, Gloom/Bassist, Crash/Drummer) and "record" an EP over one simulated week. Each day they argue in character, then a local music AI (ACE-Step) generates the actual audio track based on what they agreed on.

File Map
File	Purpose
prompts/singer.txt	Rex character prompt
prompts/guitarist.txt	Johnathan character prompt
prompts/bassist.txt	George character prompt
prompts/drummer.txt	Charles character prompt
bot.py	Bot client — connects to coordinator, runs LLM + TTS
engine.py	LLM wrapper, history windowing, response sanitization
day_arc.py	Day/session lifecycle — opens sessions, midnight injection, triggers ACE-Step
ace_step_bridge.py	Wraps ACE-Step pipeline; estimates song duration from structure/BPM
ep_state.json	All completed tracks (persists across restarts)
song_state.json	Current day's song fields
conv_log.jsonl	Full conversation log (JSONL, one line per utterance)
output/	Generated WAV files + _input_params.json sidecars
runs/	Archives of previous runs
Run History
Run 1 — archived at runs/run_20260505_233914/
Band name: not extracted (Sunday band-name day failed)
Tracks: Sunday through Thursday — only Monday produced real lyrics ("Faultline 9"). Most days the guitarist bot looped identically for the entire session, preventing lyric generation.

Run 2 — completed 2026-05-06
Band name: Sonic Anarchy

Day	Title	Notes
Sunday	Revolutionary Catharsis	Band name day
Monday	Untitled	Minimal output
Tuesday	The Last Beat of a Dying Heart	Only productive day — 11 lines of lyrics
Wednesday	Untitled	Guitarist loop again
Thursday	War of the Scales	Empty lyrics — guitarist looped all day; ACE-Step generated instrumental
Key Code Changes Made This Session
ace_step_bridge.py — Minimum 3.5-minute songs
Added _MIN_DURATION = 210 and _estimate_duration(song_doc) that computes duration from structure sections × bars × seconds-per-bar, floored at 210s. Long sections (verse, chorus, bridge) = 16 bars; short (intro, outro) = 8 bars.

day_arc.py — ~8 hour fast-mode runtime

_FAST_SESSION_MINUTES = 90      # 90 min conversation per day
_FAST_BAND_NAME_MINUTES = 20    # Sunday name lock
_FAST_MIDNIGHT_MINUTES = 90     # when midnight fires (from session open)
_FAST_SLEEP_OFFSET_MINUTES = 2
_FAST_PARTY_MINUTES = 105       # listening party (15 min gap gives ACE-Step time)
5 days × 90 min = ~7.5 hours conversation + generation time ≈ 8 hours total.


History window is only 10 messages (local_history[-10:]) — model can't see its own older responses
Old prompts gave no concrete "contribution mandate" — bots had nothing specific to add so they recycled the same vague in-character lines
repeat_penalty=1.15 handles token-level repetition but not idea-level repetition
The new prompts address this by making novelty-seeking a character identity trait, not a meta-rule.

How to Start a New Run
Archive current state (move ep_state.json, song_state.json, conv_log.jsonl, output/*.wav, output/*.json to a timestamped folder in runs/)
Delete or reset ep_state.json and song_state.json
Run the normal CLI command — the system auto-starts from Sunday since ep_state.json will be empty
The system resumes from the correct day on restart by reading ep_state.json — no manual intervention needed after a Ctrl+C.

What to Try Next
Run 3 with new prompts — observe whether the anti-sellout/glitch rules reduce looping
If looping persists: consider raising temperature slightly (currently 0.75) or widening the history window beyond 10 messages in bot.py:226
Thursday's track had empty lyrics both runs — may need a fallback lyric injection if none are extracted by midnight