import os
import sys
import time
import shutil
import math

def pretty_time(seconds):
    seconds = int(round(seconds))
    s = seconds % 60
    m = (seconds // 60) % 60
    h = (seconds // 3600)
    if h > 0:
        return '%d:%02d:%02d' % (h, m, s)
    return '%02d:%02d' % (m, s)

def format_count(n):
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return str(int(n))

class TerminalStyle:
    """Detects terminal capabilities and provides Charm-inspired color styling."""
    def __init__(self, stream=sys.stdout):
        self.stream = stream
        self.is_tty = hasattr(stream, 'isatty') and stream.isatty()
        self.supports_color = self._check_color_support()
        self.supports_unicode = self._check_unicode_support()

    def _check_color_support(self):
        if 'NO_COLOR' in os.environ:
            return False
        if os.environ.get('TERM') == 'dumb':
            return False
        if 'FORCE_COLOR' in os.environ:
            return True
        return self.is_tty

    def _check_unicode_support(self):
        if not self.is_tty:
            return False
        encoding = getattr(self.stream, 'encoding', '') or ''
        return 'utf' in encoding.lower()

    # Charm-inspired palette: muted, elegant colors
    def dim(self, s): return f"\033[2m{s}\033[0m" if self.supports_color else s
    def bold(self, s): return f"\033[1m{s}\033[0m" if self.supports_color else s
    def cyan(self, s): return f"\033[36m{s}\033[0m" if self.supports_color else s
    def magenta(self, s): return f"\033[35m{s}\033[0m" if self.supports_color else s
    def green(self, s): return f"\033[32m{s}\033[0m" if self.supports_color else s
    def yellow(self, s): return f"\033[33m{s}\033[0m" if self.supports_color else s
    def blue(self, s): return f"\033[34m{s}\033[0m" if self.supports_color else s
    def bar_fill(self, s): return f"\033[38;5;75m{s}\033[0m" if self.supports_color else s
    def bar_empty(self, s): return f"\033[38;5;238m{s}\033[0m" if self.supports_color else s

    @property
    def check_mark(self):
        return self.green("✔") if self.supports_unicode else self.green("[OK]")

    @property
    def arrow(self):
        return "▸" if self.supports_unicode else ">"

    @property
    def bullet(self):
        return "◆" if self.supports_unicode else "*"


class ProgressBar:
    """
    Modern Charm-inspired terminal progress bar.
    Provides responsive rendering, rate computation, stage labels,
    and clean fallback for non-TTY streams (CI, log files).
    """
    def __init__(
        self,
        total=100,
        start_value=0,
        label="",
        unit="it",
        enabled=True,
        stream=sys.stdout
    ):
        self.min_value = start_value
        self.max_value = max(start_value, total)
        self.value = start_value
        self.label = label
        self.unit = unit
        self.enabled = enabled
        self.stream = stream
        self.style = TerminalStyle(stream)
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.last_milestone = -1
        self._stopped = False
        self._last_line_len = 0

    @property
    def percent_complete(self):
        span = self.max_value - self.min_value
        if span <= 0:
            return 100.0
        pct = (self.value - self.min_value) / span * 100.0
        return max(0.0, min(100.0, pct))

    @property
    def elapsed_time(self):
        return time.time() - self.start_time

    @property
    def eta(self):
        t = self.percent_complete / 100.0
        if t <= 0:
            return 0
        return (1.0 - t) * self.elapsed_time / t

    @property
    def rate(self):
        dt = self.elapsed_time
        if dt <= 0:
            return 0.0
        return (self.value - self.min_value) / dt

    def increment(self, delta=1):
        self.update(self.value + delta)

    def update(self, value):
        if self._stopped:
            return
        if self.value >= self.max_value and value >= self.max_value and self._last_line_len > 0:
            return
        self.value = min(value, self.max_value)
        if not self.enabled:
            return

        now = time.time()
        if self.style.is_tty:
            # Throttle re-renders slightly to avoid terminal flicker (at most 30 FPS)
            if now - self.last_update_time > 0.033 or self.value >= self.max_value:
                self.last_update_time = now
                line = self.render()
                self._write_tty(line)
        else:
            # Non-TTY mode (e.g. CI / logs): report periodic milestones cleanly
            pct = int(self.percent_complete)
            milestone = (pct // 25) * 25
            if milestone > self.last_milestone and milestone > 0:
                self.last_milestone = milestone
                lbl = f"{self.label}: " if self.label else ""
                self.stream.write(f"  {lbl}{milestone}% ({self.value}/{self.max_value} {self.unit})\n")
                self.stream.flush()

    def _write_tty(self, line):
        pad = " " * max(0, self._last_line_len - len(line))
        self.stream.write(f"\r  {line}{pad}")
        self.stream.flush()
        self._last_line_len = len(line)

    def render_bar(self, size=24):
        pct = self.percent_complete / 100.0
        filled = int(round(pct * size))
        empty = size - filled
        if self.style.supports_unicode:
            char_fill = "━"
            char_empty = "━"
            return f"{self.style.bar_fill(char_fill * filled)}{self.style.bar_empty(char_empty * empty)}"
        else:
            return f"[{'#' * filled}{'-' * empty}]"

    def render(self):
        term_width = shutil.get_terminal_size((80, 24)).columns
        pct_str = f"{self.percent_complete:5.1f}%"
        
        val_str = f"{format_count(self.value)}/{format_count(self.max_value)} {self.unit}"
        
        rate_val = self.rate
        rate_str = f"{format_count(rate_val)} {self.unit}/s" if rate_val > 0 else ""
        time_str = f"{pretty_time(self.elapsed_time)}<{pretty_time(self.eta)}"

        prefix = f"{self.style.bold(self.label)} " if self.label else ""
        fixed_text_len = len(self.label) + len(pct_str) + len(val_str) + len(rate_str) + len(time_str) + 16
        avail = max(10, min(30, term_width - fixed_text_len))

        bar_str = self.render_bar(size=avail)
        
        parts = []
        if self.label:
            parts.append(self.style.cyan(self.label))
        parts.append(self.style.bold(pct_str))
        parts.append(bar_str)
        parts.append(self.style.dim(val_str))
        if rate_str and avail >= 15:
            parts.append(self.style.dim(rate_str))
        parts.append(self.style.dim(time_str))

        return " ".join(p for p in parts if p)

    def done(self, message=None):
        if self._stopped:
            return
        if self.value < self.max_value:
            self.update(self.max_value)
        self.stop(message=message)

    def stop(self, message=None):
        if self._stopped:
            return
        self._stopped = True
        if not self.enabled:
            return

        if self.style.is_tty:
            if message:
                self.stream.write(f"\r  {message}\033[K\n")
            else:
                self.stream.write("\n")
            self.stream.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.done()
        else:
            self.stop()


class Bar(ProgressBar):
    """
    Backward-compatible drop-in replacement for the original sdf.progress.Bar.
    """
    def __init__(self, max_value=100, min_value=0, enabled=True, label=""):
        super().__init__(
            total=max_value,
            start_value=min_value,
            label=label,
            unit="batches",
            enabled=enabled
        )

    def render_percent_complete(self):
        return '%3.0f%%' % self.percent_complete

    def render_value(self):
        if self.min_value == 0:
            return '(%g of %g)' % (self.value, self.max_value)
        return '(%g)' % self.value

    def render_bar(self, size=30):
        a = int(round(self.percent_complete / 100.0 * size))
        b = size - a
        return '[' + '#' * a + '-' * b + ']'

    def render_elapsed_time(self):
        return pretty_time(self.elapsed_time)

    def render_eta(self):
        return pretty_time(self.eta)

    def render(self):
        if not self.label:
            items = [
                self.render_percent_complete(),
                self.render_value(),
                self.render_bar(),
                self.render_elapsed_time(),
                self.render_eta(),
            ]
            return ' '.join(items)
        return super().render()
