"""Spoken feedback for the desktop demo. (The Flutter app uses flutter_tts on the phone.)"""

from __future__ import annotations

import queue
import threading
import time


class Speaker:
    def __init__(self, enabled: bool = True, cooldown: float = 8.0):
        self.cooldown = cooldown
        self._last_text = ""
        self._last_time = 0.0
        self._queue: queue.Queue[str] = queue.Queue(maxsize=1)
        self.enabled = enabled and self._engine_available()
        if self.enabled:
            threading.Thread(target=self._worker, daemon=True).start()

    @staticmethod
    def _engine_available() -> bool:
        try:
            import pyttsx3

            pyttsx3.init()
            return True
        except Exception as exc:  # no audio driver (e.g. headless server)
            print(f"[voice] disabled: {exc}")
            return False

    def say(self, text: str) -> None:
        """Speak only if the message changed or the cooldown passed; never blocks the video loop."""
        now = time.monotonic()
        if not self.enabled or (text == self._last_text and now - self._last_time < self.cooldown):
            return
        self._last_text, self._last_time = text, now
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass

    def _worker(self) -> None:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        while True:
            engine.say(self._queue.get())
            engine.runAndWait()
