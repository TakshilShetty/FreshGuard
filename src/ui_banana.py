"""PeelLab-style banana artwork + animated background layer (presentation only).

One <symbol> is defined once and re-used with <use>, so the whole flow costs a
handful of DOM nodes. Motion is pure CSS (see assets/spore_watch.css).
"""
from __future__ import annotations

import random

BANANA_SYMBOL = (
    '<symbol id="bn" viewBox="0 0 120 120">'
    '<path d="M16 22C10 62 38 108 96 96C106 94 112 88 108 82C80 92 44 78 34 22C33 17 18 15 16 22Z" '
    'fill="#FFC531" stroke="#17140D" stroke-width="4" stroke-linejoin="round"/>'
    '<path d="M31 32C41 76 78 90 103 83" fill="none" stroke="#E39B00" stroke-width="5" stroke-linecap="round"/>'
    '<path d="M22 36C24 62 37 85 58 96" fill="none" stroke="#FFE58A" stroke-width="4" stroke-linecap="round"/>'
    '<path d="M16 22C14 14 19 8 27 8L33 11C32 15 34 18 34 22Z" fill="#8A6A23" stroke="#17140D" stroke-width="3.5" stroke-linejoin="round"/>'
    '<path d="M96 96C105 94 110 89 108 82C104 85 100 86 95 87Z" fill="#3B2A10"/>'
    "</symbol>"
)

_COUNT = 16


def banana_icon(cls: str = "") -> str:
    return f'<svg class="{cls}" viewBox="0 0 120 120" aria-hidden="true" focusable="false"><use href="#bn"/></svg>'


def background_layer() -> str:
    """Fixed, click-through layer of drifting bananas. Deterministic layout."""
    rnd = random.Random(11)
    items = []
    for i in range(_COUNT):
        size = rnd.choice([34, 46, 58, 72, 92, 120])
        x, y = rnd.uniform(-8, 92), rnd.uniform(2, 92)
        dur, sw = rnd.uniform(46, 90), rnd.uniform(5, 9)
        delay = -rnd.uniform(0, dur)
        rot = rnd.uniform(-40, 40)
        dx, dy = rnd.uniform(18, 34), rnd.uniform(-14, 8)
        op = 0.55 if size < 50 else 0.9
        items.append(
            f'<i class="bn" style="--s:{size}px;--x:{x:.0f}vw;--y:{y:.0f}vh;--dur:{dur:.0f}s;--sw:{sw:.1f}s;'
            f'--d:{delay:.0f}s;--r:{rot:.0f}deg;--dx:{dx:.0f}vw;--dy:{dy:.0f}vh;--o:{op}">{banana_icon()}</i>'
        )
    return (
        '<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>' + BANANA_SYMBOL + "</defs></svg>"
        '<div class="sw-flow" aria-hidden="true">' + "".join(items) + "</div>"
    )
