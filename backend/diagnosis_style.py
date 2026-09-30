"""Diagnosis -> (colour, symbol) style map for the universe scene.

Vendored so the open backend carries no dependency on the private `ammonix`
package: the serve path needs exactly this one table and nothing else from it.
Colours are colourblind-friendly; symbols are Plotly Scatter3d filled shapes.

Keys are lower-cased diagnosis names; look up with `name.lower()`.
"""

from __future__ import annotations

DIAGNOSIS_STYLE: dict[str, tuple[str, str]] = {
    # Sinus rhythms -> Blue
    "sinus rhythm": ("#4A90D9", "circle"),
    "sinus tachycardia": ("#4A90D9", "square"),
    "sinus bradycardia": ("#4A90D9", "diamond"),
    "sinus irregularity": ("#4A90D9", "cross"),

    # Non-sinus tachyarrhythmias -> Purple
    "atrial fibrillation": ("#7B1FA2", "circle"),
    "atrial flutter": ("#7B1FA2", "cross"),
    "atrial tachycardia": ("#7B1FA2", "square"),
    "supraventricular tachycardia": ("#7B1FA2", "diamond"),
    "ventricular tachycardia": ("#7B1FA2", "cross"),

    # Axis + mild conduction -> Amber
    "normal axis": ("#F9A825", "circle"),
    "axis left shift": ("#F9A825", "square"),
    "axis right shift": ("#F9A825", "diamond"),
    "1° av-block": ("#F9A825", "cross"),

    # Higher AV block + junctional + device -> Brown
    "2° av-block": ("#6D4C41", "circle"),
    "3° av-block": ("#6D4C41", "square"),
    "junctional rhythm": ("#6D4C41", "diamond"),
    "pacing rhythm": ("#6D4C41", "cross"),

    # BBB / hypertrophy / cardiac function -> Green
    "right bundle branch block": ("#2E7D32", "circle"),
    "left bundle branch block / variations": ("#2E7D32", "square"),
    "left ventricular hypertrophy": ("#2E7D32", "diamond"),
    "lvef ≤45%": ("#2E7D32", "cross"),

    # ST / T-wave / QT -> Red
    "st deviation": ("#E53935", "circle"),
    "t wave change": ("#E53935", "square"),
    "t wave inversion": ("#E53935", "diamond"),
    "qt interval extension": ("#E53935", "cross"),

    # Ectopy / ischemia -> Dark Blue
    "atrial premature beats": ("#1A237E", "circle"),
    "ventricular premature beats": ("#1A237E", "square"),
    "myocardial infarction": ("#1A237E", "diamond"),
    "stemi": ("#1A237E", "cross"),
}

FALLBACK_COLOR = "#BDBDBD"
FALLBACK_SYMBOL = "circle"

DIAGNOSIS_COLORS: dict[str, str] = {dx: style[0] for dx, style in DIAGNOSIS_STYLE.items()}


def get_style_for_diagnosis(diagnosis: str) -> tuple[str, str]:
    """(colour, symbol) for a diagnosis name, case-insensitive."""
    return DIAGNOSIS_STYLE.get(str(diagnosis).lower().strip(), (FALLBACK_COLOR, FALLBACK_SYMBOL))


def get_color_for_diagnosis(diagnosis: str) -> str:
    return get_style_for_diagnosis(diagnosis)[0]


def get_symbol_for_diagnosis(diagnosis: str) -> str:
    return get_style_for_diagnosis(diagnosis)[1]
