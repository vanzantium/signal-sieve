"""Starter Spanish lexicon. It is not yet corpus-calibrated to English parity."""

from .base import LexiconPack


PACK = LexiconPack(
    lang_code="es",
    script="latin",
    calibration="starter_partial",
    certainty_markers=[
        "probado", "probada", "comprobado", "comprobada", "está comprobado",
        "está comprobada", "demostrado", "demostrada", "innegable",
        "obviamente", "sin duda", "definitivamente", "garantizado",
        "la verdad es", "todos saben", "claramente", "absolutamente",
        "irrefutable", "es un hecho", "100%", "es evidente", "no hay duda",
    ],
    hedge_markers=[
        "sugiere", "puede", "podría", "parece", "indica", "preliminar",
        "estimado", "aproximadamente", "probablemente", "posiblemente",
        "asociado con", "en algunos casos", "tiende a",
    ],
    hype_markers=[
        "avance", "milagro", "milagrosa", "milagroso", "impactante",
        "revolucionario", "no creerás", "increíble", "secreto",
        "no quieren que sepas", "alucinante", "cura milagrosa", "bomba",
    ],
    attack_markers=[
        "despierten", "despierta", "investiga por tu cuenta",
        "los medios mienten", "la prensa miente", "vendido", "títere",
        "lavado de cerebro", "abran los ojos", "abre los ojos",
        "lo que ocultan", "sigue el dinero",
    ],
    evidence_markers=[
        "estudio", "estudios", "datos", "metodología", "según", "investigación",
        "investigadores", "ensayo", "experimento", "muestra",
        "revisado por pares", "publicado", "hallazgos", "resultados",
        "análisis", "encuesta", "informe", "evidencia", "medido", "observado",
    ],
    anonymous_markers=[
        "fuentes dicen", "fuentes cercanas", "según se informa", "se rumorea",
        "una persona familiarizada", "allegados", "fuentes anónimas",
    ],
    month_names=[
        "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
        "agosto", "septiembre", "octubre", "noviembre", "diciembre",
    ],
).compile()
