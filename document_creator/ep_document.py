import json, os, datetime

EP_STATE_PATH = "ep_state.json"
_TOTAL_TRACKS = 10  # 5 days × 2 tracks/day — keep in sync with DayArc._TRACKS_PER_DAY


class EPDocument:

    def __init__(self):

        self.band_name = None
        self.tracks = []

    def save(self):

        data = {
            "band_name": self.band_name,
            "tracks": self.tracks,
        }
        tmp_path = EP_STATE_PATH + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                f.write(json.dumps(data, indent=2))
            os.replace(tmp_path, EP_STATE_PATH)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    @classmethod
    def load(cls):

        instance = cls()

        if not os.path.exists(EP_STATE_PATH):
            return instance

        try:
            with open(EP_STATE_PATH, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {EP_STATE_PATH}: {e}")
            return instance

        instance.band_name = data.get("band_name")
        instance.tracks = data.get("tracks", [])
        return instance

    def add_track(self, day, song_doc, variants):
        """variants: list of {"path": str, "duration_sec": float} — one per generated audio."""
        self.tracks.append({
            "day": day,
            "title": song_doc.title or "Untitled",
            "variants": variants,
            "selected_variant": 0,
            "generated_at": datetime.datetime.utcnow().isoformat(),
        })
        self.save()

    def is_set_ready(self):

        return len(self.tracks) == _TOTAL_TRACKS and all(len(t.get("variants", [])) > 0 for t in self.tracks)

    def set_summary(self):

        name = self.band_name or "?"
        n = len(self.tracks)
        total_sec = 0.0
        for t in self.tracks:
            variants = t.get("variants", [])
            idx = t.get("selected_variant", 0)
            if variants:
                total_sec += variants[min(idx, len(variants) - 1)].get("duration_sec", 0)
        minutes = round(total_sec / 60) if total_sec else 0
        return f"Band: {name} | Tracks: {n}/{_TOTAL_TRACKS} | Runtime: ~{minutes}min"
