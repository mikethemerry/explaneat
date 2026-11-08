"""Export pack: an explained genome's full explanation as JSON (canonical), LaTeX, TikZ, PDF
and DOCX. See ``docs/export_pack.md``.

``collect`` is the only module that reads the database; every renderer reads the JSON only.
"""

SCHEMA_VERSION = "1.0.0"
