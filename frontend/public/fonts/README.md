NOTE (2026-09 brand retheme): the app no longer uses the files in this folder.
The brand fonts are now Hanken Grotesk and IBM Plex Mono, still self-hosted,
bundled from the `@fontsource-variable/hanken-grotesk` and
`@fontsource/ibm-plex-mono` npm packages via `@import` in `src/index.css`.
The Inter and JetBrains Mono files below are kept on disk but are not referenced.

Self-hosted fonts (V3.27 — Q-A3-19)

Source: @fontsource-variable v5.2.8 (latin subset, variable weight axis 100-900).

| File | Family | Subset | Weights | Size |
|------|--------|--------|---------|------|
| `InterVariable.woff2` | Inter | latin | 100-900 (variable) | 47 KB |
| `JetBrainsMonoVariable.woff2` | JetBrains Mono | latin | 100-900 (variable) | 40 KB |

Both fonts are licensed under the SIL Open Font License v1.1; full text in
`Inter-LICENSE.txt` and `JetBrainsMono-LICENSE.txt`.

Replaces the prior Google Fonts CDN load — every page request was hitting
fonts.googleapis.com / fonts.gstatic.com with the WaveMedix referrer, which
is a privacy / PHI concern in clinical deployments. Self-hosting the woff2
files keeps font traffic on the WaveMedix origin.

@font-face declarations live in `src/index.css`. Updates to these files
should re-snapshot via:

    curl -sLo InterVariable.woff2 https://cdn.jsdelivr.net/npm/@fontsource-variable/inter@<version>/files/inter-latin-wght-normal.woff2
    curl -sLo JetBrainsMonoVariable.woff2 https://cdn.jsdelivr.net/npm/@fontsource-variable/jetbrains-mono@<version>/files/jetbrains-mono-latin-wght-normal.woff2
