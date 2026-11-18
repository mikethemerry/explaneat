# Schema for one experiment card. Each data file data/<anchor>.py (anchor like e1_3, dashes->underscores)
# defines CARD = {...}. Text fields use an inline mini-markup: plain text, `code` -> <code>,
# **bold** -> <strong>. Do NOT write raw HTML; the builder escapes &, <, >.
# Never write the word "Claude" in any text (use "agent-assisted" / "the explaining agent").
CARD = {
    "id": "1.3",                      # manifest number
    "anchor": "e1-3",                 # bare token for #e1-3 (letters/digits/dash)
    "group": "e1",                    # e1 | e2 | e3 | sec | side
    "name": "Cox survival case (GBSG2, WHAS500)",
    "status": "complete",             # complete | interim | gap
    "status_note": "plain-language draft deliberately skipped",   # short, from manifest
    "prereg": "exploratory",          # "pre-registered" | "exploratory" | "design fixed before run" | other short phrase
    "oneline_question": "Can a Cox-loss PropNEAT network match standard survival models and be explained?",
    "headline": "GBSG2 Harrell C 0.675 vs ridge 0.680; WHAS500 0.830 vs 0.819; verifier upheld / weakened",  # numbers copied exactly
    "question": "One or two sentences.",
    "design": [                       # list of [label, text]; use these labels where they apply, in this order
        ["Data and splits", "..."],
        ["Models and config", "population, generations, epochs, seeds ..."],
        ["Selection rule", "..."],
        ["Explanation and verifier", "..."],
        ["Metrics", "..."],
        ["Registration", "pre-registered / exploratory, with the quote and file"],
        ["Departures from plan", "..."],
    ],
    "results": [                      # small tables, numbers copied EXACTLY from the named file
        {
            "caption": "Test performance",
            "source": "experiments/survival/cox_case/tables.md",   # repo-relative path (+ section)
            "headers": ["Dataset", "Model", "Harrell C"],
            "rows": [["GBSG2", "PropNEAT-Cox", "0.675"]],
            "note": "optional one line, e.g. 'interim' or 'files disagree: X says a, Y says b'",
        },
    ],
    "results_text": ["optional bullet sentences with further key numbers, each ending with (source: path)"],
    "shows": "Two to four plain sentences, meaning first, jargon last.",
    "caveats": ["...", "..."],
    "figures": [                      # 0-2 PNGs, copied by you into ch7/figs/ (keep original resolution; builder downscales)
        {"file": "figs/e1-3_1.png", "caption": "...", "source": "experiments/.../x.png"},
    ],
    "sources": ["files (with commit if known)", "DB: explaneat_dev experiments ...", "commits ..."],
    "thesis_note": "Where it fits in Ch 7; manifest flags (e.g. numbers already reported, probe v1 vs v2, not in thesis).",
}
