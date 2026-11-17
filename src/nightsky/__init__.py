"""nightsky: configurable, physically based night-sky renderer.

Pipeline: source models (galaxy / starfield) -> TOASky (sky.py) -> atmosphere (scattering,
extinction, airglow) -> foreground (Mitsuba terrain lit by the sky) -> compose + camera
(exposure, glare, tone map). Conventions live in frames.py and spectral.py.

Run everything from the repository's src/ directory (the flat modules photometry, cluster,
companion, bright_stars, views are imported directly).
"""
