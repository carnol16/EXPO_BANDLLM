import asyncio
import dataclasses
import datetime
import json
import os
import random
import threading
import wave

import ace_step_bridge
import vocal_synthesizer
from document_creator.song_document import SongDocument
from document_creator.ep_document import EPDocument
from resource_manager import ResourceManager


def _song_duration(path) -> float:
    """Return duration in seconds via soundfile (WAV/OGG/FLAC/AIFF). Falls back to 180s."""
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return info.duration
    except Exception:
        pass
    return 180.0


@dataclasses.dataclass
class DayConfig:
    name: str
    wall_clock_start: datetime.time
    opening_injection: str
    is_band_name_day: bool


DAY_CONFIGS = [
    DayConfig(
        name="sunday",
        wall_clock_start=datetime.time(0, 0),
        opening_injection="It's Sunday. You haven't agreed on a band name yet. What are you calling this thing?",
        is_band_name_day=True,
    ),
    DayConfig(
        name="monday",
        wall_clock_start=datetime.time(0, 0),
        opening_injection="New day. New track. What's it about?",
        is_band_name_day=False,
    ),
    DayConfig(
        name="tuesday",
        wall_clock_start=datetime.time(0, 0),
        opening_injection="New day. New track. What's it about?",
        is_band_name_day=False,
    ),
    DayConfig(
        name="wednesday",
        wall_clock_start=datetime.time(0, 0),
        opening_injection="New day. New track. What's it about?",
        is_band_name_day=False,
    ),
    DayConfig(
        name="thursday",
        wall_clock_start=datetime.time(0, 0),
        opening_injection="New day. New track. What's it about?",
        is_band_name_day=False,
    ),
]


class DayArc:

    # Fast-mode timing (minutes of real time per "day")
    _FAST_SESSION_MINUTES = 10
    _FAST_BAND_NAME_MINUTES = 8
    _FAST_MIDNIGHT_MINUTES = 2
    _FAST_THREE_QUARTER_MINUTES = int(_FAST_SESSION_MINUTES * 0.75)  # 22 min
    _FAST_SLEEP_OFFSET_MINUTES = 1
    _FAST_PARTY_MINUTES = 35

    # Real-mode timing — start midnight Monday → all tracks done by Thursday 5 PM (89h).
    # 10 tracks × (6.4h talk + ~2.5h overhead/ACE-Step) = 89h → done ~Thursday 5 PM.
    _REAL_SESSION_HOURS = 6.1

    _TRACKS_PER_DAY = 2

    def __init__(
        self,
        inject_callback,
        osc_sender,
        song_document,
        ep_document,
        resource_manager,
        moderator_model_path,
        coordinator,
        bot_names,
        day_override=None,
        fast_mode=False,
    ):
        self.inject_callback = inject_callback
        self.osc_sender = osc_sender
        self.song_document = song_document
        self.ep_document = ep_document
        self.resource_manager = resource_manager
        self.moderator_model_path = moderator_model_path
        self.coordinator = coordinator
        self.bot_names = bot_names
        self.day_override = day_override
        self.fast_mode = fast_mode

        self._sleep_schedule: dict[str, datetime.datetime] = {}
        self._midnight_fired: bool = False
        self._extraction_done: bool = False
        self._band_name_extracted: bool = False
        self._three_quarter_fired: bool = False
        self._current_day_config: DayConfig | None = None
        self._session_open: bool = False
        self._session_open_time: datetime.datetime | None = None
        self._session_open_date: datetime.date | None = None
        self._listening_party_fired: bool = False
        self._thursday_ep_done: bool = False
        self._current_track_num: int = 1
        self._night_task: asyncio.Task | None = None
        self._last_extraction_time: datetime.datetime | None = None
        self._extraction_running: bool = False

    # ------------------------------------------------------------------ helpers

    def _osc(self, tag, msg=""):
        if self.osc_sender is not None:
            self.osc_sender.send_message(tag, msg)

    def _determine_current_day(self):
        if self.day_override:
            for cfg in DAY_CONFIGS:
                if cfg.name == self.day_override:
                    completed = {t["day"] for t in self.ep_document.tracks}
                    for i in range(1, self._TRACKS_PER_DAY + 1):
                        if f"{cfg.name}_{i}" not in completed:
                            self._current_track_num = i
                            return cfg
                    self._current_track_num = 1
                    return cfg
            raise ValueError(f"Unknown day override: {self.day_override!r}")

        completed = {t["day"] for t in self.ep_document.tracks}
        for cfg in DAY_CONFIGS:
            for i in range(1, self._TRACKS_PER_DAY + 1):
                if f"{cfg.name}_{i}" not in completed:
                    self._current_track_num = i
                    return cfg

        print("[DayArc] All tracks complete")
        return None

    # ------------------------------------------------------------------ session open

    def _open_session(self, day_config):
        self._session_open = True
        self._midnight_fired = False
        self._extraction_done = False
        self._three_quarter_fired = False
        self._sleep_schedule = {}
        self._session_open_time = datetime.datetime.now()
        self._session_open_date = datetime.date.today()

        # Only reset day-scoped flags on the first track of a day
        if self._current_track_num == 1:
            self._band_name_extracted = False
            self._listening_party_fired = False

        self._last_extraction_time = datetime.datetime.now()
        self.resource_manager.reset()
        self._osc("/band/day", day_config.name)
        print(f"[DayArc] Session open: {day_config.name}, track {self._current_track_num}")

        # Write fresh song_state.json immediately so the coordinator can broadcast it
        # from turn 1 (otherwise the first 10-turn summary broadcast fails with "not found")
        if self.song_document._save_path:
            try:
                self.song_document.save(self.song_document._save_path)
            except Exception as e:
                print(f"[DayArc] WARNING: could not write initial song_state.json: {e}")

        if self._current_track_num > 1:
            ep_ctx = f" {self.ep_document.set_summary()}"
            self.inject_callback(
                f"Track {self._current_track_num - 1} is recorded. Now write track {self._current_track_num}.{ep_ctx}"
            )
        elif day_config.is_band_name_day:
            self.inject_callback(day_config.opening_injection)
        else:
            ep_ctx = f" We have {len(self.ep_document.tracks)} tracks done. {self.ep_document.set_summary()}"
            self.inject_callback(day_config.opening_injection + ep_ctx)

    # ------------------------------------------------------------------ midnight

    async def _midnight_sequence(self):
        self._midnight_fired = True

        summary = self.song_document.summary()
        injection = f"You just spent all day arguing about {summary}. Session's over. Keep talking."
        self.inject_callback(injection)

        await self._extract_song_fields()
        await self._distill_acestep_caption()

        base = datetime.datetime.now()
        schedule = {}
        if self.fast_mode:
            for name in self.bot_names:
                offset_sec = random.randint(0, 30)
                schedule[name] = base + datetime.timedelta(
                    minutes=self._FAST_SLEEP_OFFSET_MINUTES, seconds=offset_sec
                )
        else:
            for name in self.bot_names:
                minutes = random.randint(5, 20)
                schedule[name] = base + datetime.timedelta(minutes=minutes)
        self._sleep_schedule = schedule
        print(f"[DayArc] Sleep schedule: { {k: v.strftime('%H:%M') for k, v in schedule.items()} }")

        if self.coordinator is not None:
            await self.coordinator.broadcast_model_swap("small")

        self._launch_acestep()

    # ------------------------------------------------------------------ extraction

    async def _extract_song_fields(self, band_name_only=False):
        try:
            with open("conv_log.jsonl", "r") as f:
                lines = f.readlines()
        except FileNotFoundError:
            print("[DayArc] conv_log.jsonl not found, skipping extraction")
            return

        recent = lines[-30:]
        transcript_parts = []
        for line in recent:
            try:
                entry = json.loads(line)
                transcript_parts.append(f"{entry.get('speaker', '?')}: {entry.get('text', '')}")
            except json.JSONDecodeError:
                continue
        transcript = "\n".join(transcript_parts)

        if self.coordinator is None:
            print("[DayArc] No coordinator — skipping LLM extraction")
            return

        try:
            if band_name_only:
                system_msg = "You are extracting a band name from a conversation. Return only valid JSON, no explanation."
                user_msg = (
                    transcript + "\n\n"
                    'What band name did they agree on? Return JSON: {"band_name": string or null}'
                )
            else:
                system_msg = "You are extracting structured song data from a conversation. Return only valid JSON, no explanation."
                user_msg = (
                    transcript + "\n\n"
                    "Extract the following fields from this conversation. Return a JSON object with these exact keys:\n"
                    "- title: string or null\n"
                    "- tempo_bpm: integer or null\n"
                    '- key: string or null (e.g. "D minor", "A major")\n'
                    "- mood: string or null\n"
                    '- lyrics: array of objects, each with "text" (string) and "author" (string, one of: singer, guitarist, bassist, drummer). Only include lines explicitly proposed as lyrics.\n'
                    '- structure: array of strings (song section names in order, e.g. ["intro", "verse", "chorus"])\n'
                    "- instruments: array of strings\n"
                    "- vibe_notes: string or null (freeform texture description)\n"
                )

            async with self.coordinator._llm_lock:
                result = await asyncio.to_thread(
                    self.coordinator.llm.create_chat_completion,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=800,
                    temperature=0.2,
                )
            raw = result["choices"][0]["message"]["content"]

        except Exception as e:
            print(f"[DayArc] Extraction LLM failed: {e}")
            return

        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip().strip("```").strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[DayArc] Extraction JSON parse failed: {e}\nRaw: {raw}")
            return

        if band_name_only:
            band_name = data.get("band_name")
            if band_name:
                self.ep_document.band_name = band_name
                self.ep_document.save()
                print(f"[DayArc] Band name extracted: {band_name}")
            self._band_name_extracted = True
            return

        for key, value in data.items():
            if value is not None:
                self.song_document.set_field(key, value)

        print(f"[DayArc] Extraction complete: {self.song_document.summary()}")
        self._extraction_done = True

    # ------------------------------------------------------------------ ACE-Step caption distillation

    async def _distill_acestep_caption(self):
        """Use the moderator LLM to write a rich ACE-Step 1.5 caption from the session summary."""
        try:
            with open("conv_log.jsonl", "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            print("[DayArc] conv_log.jsonl not found, skipping caption distillation")
            return

        # Use last 20 lines for richer context than extraction uses
        recent = lines[-20:]
        transcript_parts = []
        for line in recent:
            try:
                entry = json.loads(line)
                transcript_parts.append(f"{entry.get('speaker', '?')}: {entry.get('text', '')}")
            except json.JSONDecodeError:
                continue
        transcript = "\n".join(transcript_parts)

        # Build structured summary from what was extracted
        doc = self.song_document
        structured = []
        if doc.mood:
            structured.append(f"Mood: {doc.mood}")
        if doc.instruments:
            structured.append(f"Instruments: {', '.join(doc.instruments)}")
        if doc.vibe_notes:
            structured.append(f"Sonic texture/vibe: {doc.vibe_notes}")
        if doc.genre:
            structured.append(f"Genre: {doc.genre}")
        structured_text = "\n".join(structured) if structured else "(no structured data yet)"

        system_msg = (
            "You are a music production AI writing a sonic description for an audio generation model. "
            "Describe HOW THE MUSIC SOUNDS across multiple dimensions: style, emotion, instruments, timbre, era, production, vocals, rhythm. "
            "Be specific and technical. Repetition of key texture words strengthens the model's adherence. "
            "Never describe lyrics, imagery, or narrative. Return only the caption — no JSON, no explanation, no quotes."
        )
        user_msg = (
            f"BAND CONVERSATION EXCERPT:\n{transcript}\n\n"
            f"AGREED SONG DETAILS:\n{structured_text}\n\n"
            "Write a SONG-SPECIFIC sonic description (MAX 400 characters) for ACE-Step audio generation.\n"
            "NOTE: The genre prefix (punk-industrial hardcore, guitar/bass/drums/screamed vocals, analog production) "
            "is already prepended separately — do NOT repeat it. Focus only on what makes THIS SONG unique.\n\n"
            "SONIC DNA FOR THIS BAND: Sex Pistols rawness, Ramones buzzsaw d-beat drive, The Misfits aggressive intensity, "
            "Nirvana heavy overdriven distortion, early TOOL Undertow era dark grinding heaviness.\n\n"
            "Cover what's SPECIFIC to this track:\n"
            "- This song's particular mood/emotional texture\n"
            "- Any unusual timbre, tone, or sonic character\n"
            "- Rhythm feel or breakdown character unique to this song\n"
            "- Any specific production texture (fuzz, feedback, noise, reverb)\n"
            "Caption:"
        )

        if self.coordinator is None:
            print("[DayArc] No coordinator — skipping caption distillation")
            return

        try:
            async with self.coordinator._llm_lock:
                result = await asyncio.to_thread(
                    self.coordinator.llm.create_chat_completion,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=500,
                    temperature=0.4,
                )
        except Exception as e:
            print(f"[DayArc] Caption distillation LLM failed: {e}")
            return

        caption = result["choices"][0]["message"]["content"].strip()
        # Strip any accidental JSON wrapper or leading label
        for prefix in ('Caption:', 'caption:', '"'):
            if caption.startswith(prefix):
                caption = caption[len(prefix):].strip()
        caption = caption.strip('"').strip()
        # Hard cap — song-specific part only (genre prefix adds ~443 chars on top)
        caption = caption[:600]

        if caption:
            self.song_document.acestep_caption = caption
            print(f"[DayArc] ACE-Step caption distilled ({len(caption)} chars):\n{caption}")
        else:
            print("[DayArc] Caption distillation returned empty — will use fallback")

    # ------------------------------------------------------------------ ACE-Step

    def _launch_acestep(self):
        day_name = self._current_day_config.name if self._current_day_config else "unknown"
        track_key = f"{day_name}_{self._current_track_num}"
        song_doc = self.song_document
        osc_sender = self.osc_sender
        resource_manager = self.resource_manager

        def _run():
            try:
                with open("voice_config.json") as f:
                    voice_config = json.load(f)
                vocal_paths = vocal_synthesizer.synthesize_vocals(
                    song_doc, voice_config, track_key, "output"
                )
            except Exception as e:
                print(f"[DayArc] Vocal synthesis failed: {e}")
                vocal_paths = {"lead": None, "backing": None}

            try:
                filepaths = ace_step_bridge.generate(
                    song_doc, track_key, osc_sender, resource_manager, vocal_paths,
                    fast_mode=self.fast_mode,
                )
            except Exception as e:
                print(f"[DayArc] ACE-Step thread crashed: {e}")
                filepaths = []
            self._on_acestep_complete(filepaths)

        # Snapshot the final song state (including distilled caption) before generation.
        # Saved to song_states/{track_key}.json so every track can be re-run or adjusted later.
        try:
            os.makedirs("song_states", exist_ok=True)
            snapshot_path = os.path.join("song_states", f"{track_key}.json")
            song_doc.save(snapshot_path)
            print(f"[DayArc] Song state snapshot saved: {snapshot_path}")
        except Exception as e:
            print(f"[DayArc] WARNING: could not save song state snapshot: {e}")

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def _on_acestep_complete(self, filepaths):
        day_name = self._current_day_config.name if self._current_day_config else "unknown"
        track_key = f"{day_name}_{self._current_track_num}"

        if filepaths:
            variants = []
            for path in filepaths:
                try:
                    with wave.open(path, "rb") as wf:
                        duration = wf.getnframes() / wf.getframerate()
                except Exception as e:
                    print(f"[DayArc] Could not read duration from {path}: {e}")
                    duration = 0.0
                variants.append({"path": path, "duration_sec": duration})
            self.ep_document.add_track(track_key, self.song_document, variants)
            print(f"[DayArc] EP updated: {self.ep_document.set_summary()}")
        else:
            print("[DayArc] ACE-Step generation failed — track not added to EP")

        if self.coordinator is not None and self.coordinator.loop is not None:
            asyncio.run_coroutine_threadsafe(self._advance(), self.coordinator.loop)

    async def _night_playlist(self):
        """Load and play random songs from run_songs/ while all bots sleep."""
        import pathlib
        import sounddevice as sd
        import soundfile as sf

        run_songs_dir = pathlib.Path("run_songs")
        if not run_songs_dir.is_dir():
            print("[DayArc] run_songs/ not found — skipping night playlist")
            return
        supported = {".wav", ".ogg", ".flac", ".aiff", ".aif"}
        songs = [p for p in run_songs_dir.iterdir() if p.suffix.lower() in supported]
        if not songs:
            print("[DayArc] run_songs/ is empty — skipping night playlist")
            return

        random.shuffle(songs)
        idx = 0
        print(f"[DayArc] Night playlist starting — {len(songs)} song(s) in run_songs/")
        self._osc("/band/night_playlist_start", str(len(songs)))

        try:
            while True:
                song = songs[idx % len(songs)]
                idx += 1
                if idx > 0 and idx % len(songs) == 0:
                    random.shuffle(songs)
                try:
                    data, samplerate = await asyncio.to_thread(sf.read, str(song), dtype="float32")
                    duration = len(data) / samplerate
                    print(f"[DayArc] Night song: {song.name} ({duration:.0f}s)")
                    self._osc("/band/night_song", str(song))
                    sd.play(data, samplerate)
                    await asyncio.sleep(duration)
                    sd.stop()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    print(f"[DayArc] Night song error ({song.name}): {e} — skipping")
                    await asyncio.sleep(2)
        except asyncio.CancelledError:
            sd.stop()
            raise

    async def _advance(self):
        if self._night_task is not None:
            self._night_task.cancel()
            try:
                await self._night_task
            except asyncio.CancelledError:
                pass
            self._night_task = None
            self._midnight_fired = False  # prevent run loop from restarting playlist before _open_session
            self._osc("/band/night_playlist_stop", "")

        day_name = self._current_day_config.name if self._current_day_config else "unknown"
        if self._current_track_num < self._TRACKS_PER_DAY:
            next_num = self._current_track_num + 1
            print(f"[DayArc] Track {self._current_track_num}/{self._TRACKS_PER_DAY} for {day_name} done — starting track {next_num}")
            self._current_track_num = next_num
            self.song_document = SongDocument()
            self.song_document._save_path = "song_state.json"
            if self.coordinator is not None:
                for name in self.bot_names:
                    await self.coordinator.set_bot_wake(name)
                await self.coordinator.broadcast_model_swap("main")
            self._open_session(self._current_day_config)
            return
        await self._advance_to_next_day()

    async def _advance_to_next_day(self):
        day_name = self._current_day_config.name if self._current_day_config else "unknown"
        print(f"[DayArc] Day {day_name} complete — advancing")

        if day_name == "thursday":
            if not self._listening_party_fired:
                if self.fast_mode:
                    await self._run_listening_party()
                else:
                    # Wake bots to fill the space until the 7 PM listening party
                    if self.coordinator is not None:
                        for name in self.bot_names:
                            await self.coordinator.set_bot_wake(name)
                        await self.coordinator.broadcast_model_swap("main")
                    self._thursday_ep_done = True
                    self._midnight_fired = True  # prevent midnight re-firing on the next poll
                    self.inject_callback(
                        "The EP is done. Show starts at 7. Keep talking until then."
                    )
            return

        # Wake all sleeping bots and swap back to full model
        if self.coordinator is not None:
            for name in self.bot_names:
                await self.coordinator.set_bot_wake(name)
            await self.coordinator.broadcast_model_swap("main")

        # Fresh song document for the new day
        self.song_document = SongDocument()
        self.song_document._save_path = "song_state.json"

        # Determine next day (EP now has the completed track)
        self._current_day_config = self._determine_current_day()
        if self._current_day_config is None:
            print("[DayArc] All days complete")
            return

        self._open_session(self._current_day_config)

    # ------------------------------------------------------------------ sleep scheduling

    async def _check_sleep_schedule(self):
        now = datetime.datetime.now()
        triggered = []
        for name, wake_time in self._sleep_schedule.items():
            if now >= wake_time:
                if self.coordinator is not None:
                    await self.coordinator.set_bot_sleep(name)
                self.resource_manager.on_bot_sleep(name)
                triggered.append(name)
        for name in triggered:
            del self._sleep_schedule[name]

    def _build_decision_injection(self):
        doc = self.song_document
        missing = []
        if not doc.title:
            missing.append("a song title")
        if not doc.tempo_bpm:
            missing.append("a BPM")
        if not doc.key:
            missing.append("a key")
        if not doc.mood:
            missing.append("a mood/vibe")
        if not doc.lyrics:
            missing.append("at least one lyric line")
        if not doc.structure:
            missing.append("a song structure (intro, verse, chorus, etc.)")
        if not doc.instruments:
            missing.append("instrument list")
        if not doc.vibe_notes:
            missing.append("sonic texture notes")

        if not missing:
            return "Clock's running out. Everything's locked — make sure you all agree on it."

        missing_str = ", ".join(missing)
        return (
            f"Session ends soon. You STILL haven't locked in: {missing_str}. "
            f"Stop arguing about everything else and decide these right now or the track ships with holes."
        )

    async def _check_band_name_timer(self):
        if self._session_open_time is None:
            return
        elapsed = datetime.datetime.now() - self._session_open_time
        threshold_min = self._FAST_BAND_NAME_MINUTES if self.fast_mode else 30
        if elapsed >= datetime.timedelta(minutes=threshold_min) and not self._band_name_extracted:
            self._band_name_extracted = True  # guard first — prevents retry loop on extraction failure
            self.inject_callback("Name's locked. Now — what's the first track about?")
            await self._extract_song_fields(True)

    # ------------------------------------------------------------------ listening party

    async def _run_listening_party(self):
        if not self.ep_document.is_set_ready():
            print("[DayArc] Listening party deferred — EP not ready yet (waiting for track generation)")
            return

        self._listening_party_fired = True

        if self.coordinator is not None:
            self.coordinator.pause_gate.set()

        self._osc("/band/listening_party", "")

        for i, track in enumerate(self.ep_document.tracks):
            variants = track.get("variants", [])
            selected_idx = track.get("selected_variant", 0)
            if variants:
                selected = variants[min(selected_idx, len(variants) - 1)]
                duration = selected.get("duration_sec", 0)
                self._osc("/band/track_path", selected.get("path", ""))
            else:
                duration = track.get("duration_sec", 0)
            self._osc("/band/track_playing", str(i + 1))
            if self.coordinator is not None:
                await self.coordinator.set_mode("performance")
            self.inject_callback(f"Track {i + 1}: {track['title']}. Say something.")
            await asyncio.sleep(duration + 150)

        if self.coordinator is not None:
            self.coordinator.pause_gate.clear()

        self.inject_callback("That's the set. We made something.")

    # ------------------------------------------------------------------ main loop

    async def run(self):
        self._current_day_config = self._determine_current_day()
        if self._current_day_config is None:
            return

        poll_interval = 5 if self.fast_mode else 30

        while True:
            await asyncio.sleep(poll_interval)
            now_time = datetime.datetime.now().time()
            elapsed = (
                datetime.datetime.now() - self._session_open_time
                if self._session_open_time is not None
                else datetime.timedelta(0)
            )

            # 1. Session open
            if not self._session_open:
                if self.fast_mode or now_time >= self._current_day_config.wall_clock_start:
                    self._open_session(self._current_day_config)

            # 2. Sunday band name timer
            if self._session_open and self._current_day_config.is_band_name_day and not self._band_name_extracted:
                await self._check_band_name_timer()

            # 3. Three-quarter check — force decisions on any missing song state fields
            if self._session_open and not self._midnight_fired and not self._three_quarter_fired:
                if self.fast_mode:
                    three_q_due = elapsed >= datetime.timedelta(minutes=self._FAST_THREE_QUARTER_MINUTES)
                else:
                    three_q_due = elapsed >= datetime.timedelta(hours=self._REAL_SESSION_HOURS * 0.75)
                if three_q_due:
                    self._three_quarter_fired = True
                    injection = self._build_decision_injection()
                    self.inject_callback(injection)
                    print(f"[DayArc] Three-quarter injection: {injection}")

            # 4. Periodic extraction — build song_state.json live throughout the session
            if self._session_open and not self._midnight_fired and not self._extraction_running:
                extract_interval_sec = 120 if self.fast_mode else 25 * 60
                last = self._last_extraction_time
                if last is None or (datetime.datetime.now() - last).total_seconds() >= extract_interval_sec:
                    self._last_extraction_time = datetime.datetime.now()
                    self._extraction_running = True
                    async def _extraction_task():
                        try:
                            await self._extract_song_fields()
                        finally:
                            self._extraction_running = False
                    asyncio.create_task(_extraction_task())

            # 5. Midnight check
            if self._session_open and not self._midnight_fired:
                if self.fast_mode:
                    midnight_due = elapsed >= datetime.timedelta(minutes=self._FAST_MIDNIGHT_MINUTES)
                else:
                    midnight_due = elapsed >= datetime.timedelta(hours=self._REAL_SESSION_HOURS)
                if midnight_due:
                    await self._midnight_sequence()

            # 6. Sleep schedule + night playlist
            if self._midnight_fired and self._sleep_schedule:
                await self._check_sleep_schedule()
            if self._midnight_fired and not self._sleep_schedule and self._night_task is None and not self._thursday_ep_done:
                self._night_task = asyncio.create_task(self._night_playlist())

            # 7. Thursday listening party
            if self._current_day_config.name == "thursday" and not self._listening_party_fired:
                if self.fast_mode:
                    party_due = elapsed >= datetime.timedelta(minutes=self._FAST_PARTY_MINUTES)
                else:
                    party_due = now_time >= datetime.time(19, 0)
                if party_due:
                    await self._run_listening_party()
