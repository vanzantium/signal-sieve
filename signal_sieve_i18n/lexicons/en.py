"""Conservative English reference lexicon."""

from .base import LexiconPack


PACK = LexiconPack(
    lang_code="en",
    script="latin",
    calibration="reference",
    certainty_markers=[
        "proven", "proof", "undeniable", "obviously", "without a doubt",
        "definitely", "guaranteed", "the truth is", "fact is", "everyone knows",
        "clearly", "100%", "absolutely", "irrefutable", "certain",
    ],
    hedge_markers=[
        "suggests", "may", "might", "could", "appears to", "indicates",
        "preliminary", "estimated", "approximately", "likely", "possibly",
        "associated with", "correlated", "in some cases",
    ],
    hype_markers=[
        "breakthrough", "miracle", "shocking", "stunning", "game-changer",
        "revolutionary", "you won't believe", "secret",
        "they don't want you to know", "exposed", "bombshell",
    ],
    attack_markers=[
        "sheeple", "wake up", "mainstream media lies", "paid shill", "puppet",
        "brainwashed", "open your eyes", "what they're hiding",
    ],
    evidence_markers=[
        "study", "studies", "data", "dataset", "methodology", "method",
        "according to", "research", "researchers", "trial", "experiment",
        "sample size", "peer-reviewed", "published", "findings", "results",
        "analysis", "survey", "report", "evidence", "measured", "observed",
    ],
    anonymous_markers=[
        "sources say", "sources familiar", "insiders", "people close to",
        "anonymous source", "a person familiar", "those briefed", "reportedly",
    ],
    month_names=[
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
        "oct", "nov", "dec",
    ],
).compile()
