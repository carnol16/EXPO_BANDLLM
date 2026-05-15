import json, os, time, datetime

class SongDocument:

    _WRITABLE_FIELDS = {"title", "tempo_bpm", "key", "mood", "lyrics", "structure", "instruments", "vibe_notes"}

    def __init__(self):

        self.title = None
        self.genre = 'deathcore'
        self.tempo_bpm = None
        self.key = None
        self.mood = None
        self.lyrics = []
        self.structure = []
        self.instruments = []
        self.vibe_notes = None
        self.acestep_caption = None  # LLM-distilled ACE-Step 1.5 caption (set at midnight)
        self.agreed_at = {}
        self._callbacks = []
        self._save_path = None
    
    def register_callback(self, fn):

        self._callbacks.append(fn)

    def set_field(self, key, value):

        if key not in self._WRITABLE_FIELDS:
            print(f"Warning: '{key}' is not a writable field, ignoring.")
            return

        setattr(self, key, value)
        self.agreed_at[key] = datetime.datetime.utcnow().isoformat()

        if self._save_path is None:
            print("Warning: _save_path not set, skipping save.")
        else:
            self.save(self._save_path)

        for fn in self._callbacks:
            fn()

    def save(self, path):

        self._save_path = path
        data = {
            "title": self.title,
            "genre": self.genre,
            "tempo_bpm": self.tempo_bpm,
            "key": self.key,
            "mood": self.mood,
            "lyrics": self.lyrics,
            "structure": self.structure,
            "instruments": self.instruments,
            "vibe_notes": self.vibe_notes,
            "acestep_caption": self.acestep_caption,
            "agreed_at": self.agreed_at,
        }
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                f.write(json.dumps(data, indent=2))
            for attempt in range(5):
                try:
                    os.replace(tmp_path, path)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.05 * (attempt + 1))
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    @classmethod
    def load(cls, path):

        instance = cls()
        instance._save_path = path

        if not os.path.exists(path):
            return instance

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return instance

        for field in ("title", "genre", "tempo_bpm", "key", "mood", "lyrics", "structure", "instruments", "vibe_notes", "acestep_caption", "agreed_at"):
            if field in data:
                setattr(instance, field, data[field])

        return instance

    def summary(self):

        title = self.title or "?"
        tempo = f"{self.tempo_bpm}BPM" if self.tempo_bpm is not None else "?BPM"
        key = self.key or "?"
        n = len(self.lyrics)
        s = f"Title: {title} | Tempo: {tempo} | Key: {key} | Lyrics: {n} lines"
        return s[:200]

    def is_complete(self):

        return self.title is not None and self.tempo_bpm is not None and len(self.lyrics) >= 1

    def to_acestep_prompt(self):

        lines = [f"grunge, deathcore, distorted guitars, d-beat, industrial drums, shouted vocals, {self.genre}"]
        if self.tempo_bpm is not None:
            lines.append(f"bpm: {self.tempo_bpm}")
        if self.key is not None:
            lines.append(f"key: {self.key}")
        if self.mood is not None:
            lines.append(f"mood: {self.mood}")
        if self.vibe_notes is not None:
            lines.append(f"vibe: {self.vibe_notes}")
        return "\n".join(lines)

    def to_acestep_lyrics(self):

        singer_lines = [l["text"] for l in self.lyrics if l.get("author") == "singer"]
        if not singer_lines:
            singer_lines = [l["text"] for l in self.lyrics]
        if not singer_lines:
            return "[Instrumental]:"

        LYRIC_SECTIONS = {"verse", "chorus", "pre-chorus", "bridge", "breakdown", "hook", "refrain"}

        # Section tag with ACE-Step qualifier hints — one qualifier max per docs
        TAG_MAP = {
            "intro":       ("Intro",         "aggressive"),
            "verse":       ("Verse",         "aggressive"),
            "pre-chorus":  ("Pre-Chorus",    "building energy"),
            "chorus":      ("Chorus",        "explosive"),
            "bridge":      ("Bridge",        "aggressive"),
            "breakdown":   ("Breakdown",     "crushing"),
            "hook":        ("Chorus",        "explosive"),
            "refrain":     ("Chorus",        "explosive"),
            "outro":       ("Outro",         None),
            "instrumental":("Instrumental",  None),
            "solo":        ("Guitar Solo",   None),
            "interlude":   ("Instrumental",  None),
        }

        # Sections where we uppercase lyrics to signal screaming intensity
        HIGH_INTENSITY = {"chorus", "hook", "refrain", "breakdown"}

        if not self.structure:
            return "[screamed vocal]\n\n[Verse - aggressive]\n" + "\n".join(singer_lines)

        lyric_positions = [i for i, s in enumerate(self.structure) if s in LYRIC_SECTIONS]
        if not lyric_positions:
            return "[screamed vocal]\n\n[Verse - aggressive]\n" + "\n".join(singer_lines)

        n = len(lyric_positions)
        base = len(singer_lines) // n
        remainder = len(singer_lines) % n

        allocations = {}
        cursor = 0
        for i, pos in enumerate(lyric_positions):
            count = base + (remainder if i == n - 1 else 0)
            allocations[pos] = singer_lines[cursor:cursor + count]
            cursor += count

        result = ["[screamed vocal]", ""]  # global vocal style tag at top

        section_counts = {}
        for i, section in enumerate(self.structure):
            base_tag, qualifier = TAG_MAP.get(section, (section.title(), None))

            if section in LYRIC_SECTIONS:
                section_counts[base_tag] = section_counts.get(base_tag, 0) + 1
                total = sum(1 for s in self.structure if s == section)
                numbered = section_counts[base_tag] > 1 or total > 1
                display_tag = f"{base_tag} {section_counts[base_tag]}" if numbered else base_tag
            else:
                display_tag = base_tag

            tag_str = f"[{display_tag} - {qualifier}]" if qualifier else f"[{display_tag}]"
            result.append(tag_str)

            lines = allocations.get(i, [])
            if section in HIGH_INTENSITY:
                lines = [l.upper() for l in lines]
            result.extend(lines)
            result.append("")

        if self.structure and self.structure[-1] == "outro":
            result.append("[Fade Out]")

        return "\n".join(result).strip()

