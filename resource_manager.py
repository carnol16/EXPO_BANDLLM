import os


class ResourceManager:

    def __init__(self, bot_names, on_full_handoff=None):

        self._bot_names = list(bot_names)
        self._sleeping_bots = set()
        self._acestep_process = None
        self._current_priority = "normal"
        self._on_full_handoff = on_full_handoff

    def register_acestep_process(self, process):

        self._acestep_process = process

    def set_acestep_priority(self, level):

        self._current_priority = level

        if self._acestep_process is None:
            return

        if self._acestep_process.poll() is not None:
            print(f"[ResourceManager] Warning: ACE-Step process already terminated, skipping priority change")
            return

        nice_value = 10 if level == "low" else -5
        try:
            os.setpriority(os.PRIO_PROCESS, self._acestep_process.pid, nice_value)
        except AttributeError:
            try:
                import psutil
                proc = psutil.Process(self._acestep_process.pid)
                if level == "low":
                    proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                else:
                    proc.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
            except Exception:
                print("[ResourceManager] Warning: process priority management not supported on this platform")

        print(f"[ResourceManager] ACE-Step priority set to: {level}")

    def on_bot_sleep(self, name):

        self._sleeping_bots.add(name)
        print(f"[ResourceManager] {name} is sleeping. Sleeping: {len(self._sleeping_bots)}/{len(self._bot_names)}")

        if self._sleeping_bots >= set(self._bot_names):
            print("[ResourceManager] All bots asleep — handing full GPU to ACE-Step")
            self.set_acestep_priority("full")
            if self._on_full_handoff is not None:
                self._on_full_handoff()

    def on_bot_wake(self, name):

        self._sleeping_bots.discard(name)
        print(f"[ResourceManager] {name} is awake.")

    def reset(self):

        self._sleeping_bots = set()
        self._current_priority = "normal"
        self._acestep_process = None
        print("[ResourceManager] Reset — new day state cleared")

    def all_bots_asleep(self):

        return len(self._bot_names) > 0 and self._sleeping_bots >= set(self._bot_names)

    def sleeping_count(self):

        return len(self._sleeping_bots)
