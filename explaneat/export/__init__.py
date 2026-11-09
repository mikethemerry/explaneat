"""Export pack: an explained genome's full explanation as JSON (canonical), LaTeX, TikZ, PDF
and DOCX. See ``docs/export_pack.md``.

``collect`` is the only module that reads the database; every renderer reads the JSON only.
"""

SCHEMA_VERSION = "1.4.0"   # 1.1: plain_language section; 1.2: layer v2 (rule claims); 1.3: shape on claims; 1.4: structure claims
