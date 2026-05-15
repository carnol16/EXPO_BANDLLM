
# ACE-Step 1.5 — Prompt Structure & Input Guide

## Overview

ACE-Step 1.5 has **two primary text inputs** plus metadata parameters. Each has a distinct job:

| Input | Role | Limit |
|-------|------|-------|
| `caption` | Overall portrait — style, mood, instruments, timbre | **512 characters max** |
| `lyrics` | Temporal script — structure, lyric content, energy flow | No hard limit |
| Metadata | Precise musical parameters (BPM, key, etc.) | Optional |

---

## 1. `caption` — The Overall Portrait

Describes the *static* qualities of the music. Supports comma-separated tags or natural language.

> ⚠️ **Do NOT include BPM, key, or tempo info here** — use the dedicated metadata parameters instead. Caption should only describe style, emotion, instruments, and timbre.

### Caption Dimensions

| Dimension | Examples |
|-----------|----------|
| **Genre/Style** | `pop, rock, jazz, lo-fi, synthwave, hip-hop, bossa nova, folk` |
| **Emotion/Atmosphere** | `melancholic, uplifting, energetic, dreamy, dark, nostalgic, euphoric` |
| **Instruments** | `acoustic guitar, piano, 808 drums, strings, synth pads, electric bass` |
| **Timbre/Texture** | `warm, crisp, airy, punchy, lush, raw, polished` |
| **Era Reference** | `80s synth-pop, 90s grunge, vintage soul, 2010s EDM` |
| **Vocal** | `female vocal, male vocal, breathy, raspy, powerful, falsetto, choir` |
| **Production Style** | `lo-fi, studio-polished, bedroom pop, live recording, high-fidelity` |
| **Speed/Feel** | `slow tempo, mid-tempo, fast-paced, groovy, driving, laid-back` |

### Caption Tips

- **Specific beats vague** — `"sad piano ballad with female breathy vocal"` works better than `"a sad song"`
- **Combine multiple dimensions** — style + emotion + instruments + timbre anchors results precisely
- **Use era/artist references** — `"in the style of 80s synthwave"` conveys complex aesthetic quickly
- **Texture adjectives matter** — `warm, crisp, airy, punchy` influence mixing and timbre
- **Avoid conflicting styles** — `"classical strings"` + `"hardcore metal"` in the same caption confuses the model
  - Fix: use temporal evolution in lyrics instead — `"Start with soft strings, build into metal, end in hip-hop"`

---

## 2. `lyrics` — The Temporal Script

Controls how the music **unfolds over time**. Carries:
- Lyric text
- Structure tags (sections)
- Vocal style hints
- Instrumental sections
- Energy changes

> ⚠️ **Caption and Lyrics must be consistent.** If Caption says `"violin solo"` but Lyrics says `[Guitar Solo]`, the model gets confused and quality drops.

### Structure Tags

```
[Intro]
[Verse] / [Verse 1] / [Verse 2]
[Pre-Chorus]
[Chorus]
[Bridge]
[Outro]
[Instrumental]
[Build]
[Drop]
[Breakdown]
[Guitar Solo]
[Piano Interlude]
[Fade Out]
[Silence]
```

**Combine tags with a modifier for finer control:**
```
[Chorus - anthemic]
[Bridge - whispered]
[Verse 1 - raspy vocal]
[Intro - ambient]
```

> ⚠️ Don't stack too many modifiers: `[Chorus - anthemic - stacked - high energy - epic]` risks the model singing the tag text. Keep it to one modifier max.

### Vocal Control Tags (inline)

| Tag | Effect |
|-----|--------|
| `[raspy vocal]` | Raspy, textured vocals |
| `[whispered]` | Whispered delivery |
| `[falsetto]` | Falsetto register |
| `[powerful belting]` | High, powerful singing |
| `[spoken word]` | Rap or recitation |
| `[harmonies]` | Layered harmonies |
| `[call and response]` | Call and response structure |
| `[ad-lib]` | Improvised embellishments |

### Energy Tags (inline)

| Tag | Effect |
|-----|--------|
| `[high energy]` | Passionate, intense |
| `[low energy]` | Restrained, quiet |
| `[building energy]` | Gradually rising |
| `[explosive]` | Sudden energy release |
| `[melancholic]` | Sad, reflective |
| `[euphoric]` | Joyful, triumphant |
| `[dreamy]` | Soft, hazy |
| `[aggressive]` | Hard, forceful |

### Lyric Writing Rules

- **6–10 syllables per line** — lines with wildly different syllable counts cause rhythm problems
- **Uppercase = intensity** — `WE ARE THE CHAMPIONS!` signals loud/shouted delivery
- **Parentheses = background vocals** — `We rise together (together)` creates harmonies
- **Blank lines between sections** — always separate structure blocks with an empty line
- **Instrumental music** — use `[Instrumental]` or describe sections with tags only (no lyric text)

---

## 3. Metadata Parameters (Optional)

Let the LM auto-infer these unless you have a specific need. When `thinking` mode is on, the model handles this automatically.

| Parameter | Range | Notes |
|-----------|-------|-------|
| `bpm` | 30–300 | Slow: 60–80 · Mid: 90–120 · Fast: 130–180 |
| `keyscale` | e.g. `C Major`, `Am`, `F# Minor` | Minor = darker · Major = brighter |
| `timesignature` | `4/4`, `3/4`, `6/8` | `4/4` most stable; `3/4` = waltz |
| `vocal_language` | 50+ languages | Auto-detected from lyrics if not set |
| `duration` | 10–600 seconds | 30–240s most stable; very long may repeat |

> ⚠️ Watch for conflicts: Caption says `"slow ballad"` but `bpm=160` = confused output. Keep them consistent.

---

## 4. Inference / Quality Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `lm_temperature` | 0.85 | 0–2.0 · Higher = more creative/random |
| `lm_cfg_scale` | 2.0 | Higher = stronger prompt adherence |
| `guidance_scale` | 7.0 | CFG for audio (SFT/Base models only) |
| `lm_negative_prompt` | — | What to avoid: `"distorted, off-key, low quality"` |
| `thinking` | True | Enables chain-of-thought planning (recommended) |
| `lm_top_p` | 0.9 | Nucleus sampling (0.9–0.95 typical) |

---

## 5. Mapping a Conversation Summary → Prompt

Since caption is limited to **512 characters**, you need to distill. Use this workflow:

### Step 1 — Extract from your summary
- What is the **emotional arc**? (tense → resolved, sad → hopeful)
- What is the **setting/vibe**? (late night city, triumphant stadium, quiet room)
- What **characters or narrative** maps to vocal style or energy?
- What is the **desired tempo feel**?

### Step 2 — Write `caption` (≤512 chars)
Compress into the most impactful descriptors. The LM's chain-of-thought will expand it internally.

### Step 3 — Write `lyrics` with structure tags
Put the narrative/story as lyric content, with tags guiding the energy arc through the song.

### Step 4 — Set metadata only if needed
If the summary implies a waltz → `3/4`. Sorrowful and slow → `bpm: 65`. Otherwise, leave it to the LM.

---

## 6. Full Examples

### Example A — Tense Thriller / Betrayal Confrontation

> *Summary: Two characters, one confronting the other about a betrayal. Tension builds, then silence, then a cold resolution.*

```
caption:
dark cinematic, male baritone vocal, minor key, tense strings,
sparse piano, cold electronic undertones, film noir atmosphere,
suspenseful, dramatic, intense, building dread

lyrics:
[Intro - sparse piano]

[Verse 1 - spoken word]
I know what you did
Every word, every lie
You thought the dark would hide you
But I've seen through disguise

[Build]

[Chorus - powerful]
HERE WE ARE NOW
Face to face with the truth
Everything you buried
Is standing in this room

[Breakdown - whispered]
There's nothing left to say now

[Outro - cold, minimal]

bpm: 75
keyscale: D Minor
timesignature: 4/4
```

---

### Example B — Personal Growth / Triumphant Recovery

> *Summary: Someone reflecting on a hard year, finding inner strength, ready to move forward. Hopeful and triumphant.*

```
caption:
uplifting pop ballad, female vocal, acoustic guitar, strings swelling,
emotional journey, intimate to epic, hopeful, resilient,
warm production, cathartic, powerful chorus

lyrics:
[Intro - acoustic guitar]

[Verse 1]
I walked through the darkest winter
Thought the cold would never end
Counted every broken promise
Learned to heal without a friend

[Pre-Chorus - building energy]
But somewhere in the silence
I heard my own voice again

[Chorus - powerful]
I am still standing
After everything I've been through
Rising from the ashes
Finally becoming new

[Bridge - whispered]
All those nights I thought I'd lost myself
Were leading me right here

[Final Chorus - euphoric]
I AM STILL STANDING
AFTER EVERYTHING I'VE BEEN THROUGH

[Outro - fade out]

bpm: 92
keyscale: G Major
timesignature: 4/4
```

---

### Example C — Pure Instrumental / Cinematic Journey

> *Summary: A long conversation about exploration and discovery — no lyrics needed, just evolving instrumental mood.*

```
caption:
cinematic orchestral, epic, sweeping strings, adventurous,
building tension, triumphant resolution, film score style,
layered textures, dynamic range, emotional depth

lyrics:
[Intro - ambient, sparse strings]

[Main Theme - building]

[Climax - full orchestra, powerful]

[Resolution - warm, hopeful]

[Outro - fade out]

bpm: 88
keyscale: E Minor
timesignature: 4/4
duration: 120
```

---

## Quick Reference Checklist

Before generating, verify:

- [ ] Caption is under 512 characters
- [ ] Caption does NOT include BPM, key, or time signature info
- [ ] Lyrics sections are separated by blank lines
- [ ] Tag modifiers are concise (one word max per tag)
- [ ] Caption mood/instruments match Lyrics section tags (no conflicts)
- [ ] Syllable count per lyric line is roughly 6–10
- [ ] Metadata parameters don't contradict Caption (e.g. no `bpm=160` + "slow ballad")
- [ ] `lm_negative_prompt` set if you want to avoid something specific
