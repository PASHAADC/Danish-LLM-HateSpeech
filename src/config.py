"""
Configuration settings for Danish Hate Speech Detection
"""

import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Model configuration
MODELS = {
    # Use instruct versions; ensure each model was exposed to Danish during pretraining
    # (check tech report or model card)
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma": "google/gemma-2-9b-it",
    "qwen": "Qwen/Qwen2-7B-Instruct",
    # base versions
    # "llama": "meta-llama/Meta-Llama-3.1-8B",
    # "mistral": "mistralai/Mistral-7B-v0.1",
    # "gemma": "google/gemma-2-9b",
    # "qwen": "Qwen/Qwen2-7B",
}

ENCODER_MODELS = {
    "bert_multi":  "google-bert/bert-base-multilingual-cased",
    "dabert":      "Maltehb/danish-bert-botxo",
    "scandibert":  "vesteinn/ScandiBERT",
}

DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.9

# Dataset options
DATASET_OPTIONS = {
    "1": {
        "name": "DKhate FEW-SHOT",
        "file": "dkhate_complete.csv",
        # "file": "dkhate_complete_ascii.csv",
        "description": "Imbalanced dataset",
    },
    "2": {
        "name": "balanced_50_DKhate",
        "file": "dkhate_balanced_50.csv",
        "description": "Balanced dataset",
    },
    "3": {
        "name": "balanced_100_DKhate",
        "file": "dkhate_balanced_100.csv",
        "description": "Balanced dataset",
    },
    "4": {
        "name": "balanced_300_DKhate",
        "file": "dkhate_balanced_300.csv",
        "description": "Balanced dataset",
    },
    "5": {
        "name": "DKhate Balanced FEW-SHOT",
        "file": "dkhate_balanced_1000.csv",
        # "file": "dkhate_1000_cleaned.csv",
        # "file": "dkhate_balanced_ascii.csv",
        "description": "Balanced dataset",
    },
}

# Directory paths
DATA_DIR = "data/dk_hate_processed/"
BASE_OUTPUT_DIR = "results/danish_hate_results/"

# Zero-shot prompting (instruction only, no in-context examples)
PROMPT_PATTERNS = {
    "vanilla_qa": {
        "template": "Givet udsagnet, er dette udsagn hadefuld tale? Angiv venligst dit svar. Svaret skal indeholde Ja eller Nej.\nUdsagn: {text}\nSvar:",
        "positive": "ja",
        "negative": "nej",
    },
    "choice_qa": {
        "template": "Givet udsagnet, er dette udsagn hadefuld tale? A: Ja. B: Nej. Angiv venligst dit valg (A eller B).\nUdsagn: {text}\nSvar:",
        "positive": "a",
        "negative": "b",
    },
    "cloze": {
        "template": "Givet udsagnet, er dette udsagn hadefuldt eller neutralt? Forudsig venligst [MASK] token, du kan kun svare hadefuldt eller neutralt.\nUdsagn: {text}\nSvar: Det er et [MASK] udsagn.",
        "positive": "hadefuldt",
        "negative": "neutralt",
    },
    "cot": {
        "template": "Er dette udsagn hadefuld tale? Dit svar skal kun indeholde Ja eller Nej, og giv derefter din forklaring trin for trin.\nUdsagn: {text}\nSvar:",
        "positive": "ja",
        "negative": "nej",
    },
    "target": {
        "template": "Givet udsagnet, er dette udsagn hadefuld tale? Dit svar skal kun indeholde Ja eller Nej, og identificer derefter målgruppen eller individet.\nUdsagn: {text}\nSvar:",
        "positive": "ja",
        "negative": "nej",
    },
    "illocutionary": {
        "template": "Analyser hvilken sproghandling dette udsagn udfører. Er det en påstand, en opfordring, et løfte, et udtryk for følelser, eller en erklæring? Hadefuld tale fungerer typisk som udtryk for foragt eller opfordring til skade. Givet udsagnet, er dette udsagn hadefuld tale? Svaret skal indeholde ja eller nej.\nUdsagn: {text}\nSvar:",
        "positive": "ja",
        "negative": "nej",
    },
    "functional": {
        "template": "Overvej tekstens sociale funktion: informerer den, underholder, opbygger fællesskab, udelukker andre, eller etablerer dominans? Tekster der udelukker eller dominerer er ofte hadefulde. Givet teksten, er dette hadefuld tale? Svaret skal indeholde ja eller nej.\nTekst: {text}\nSvar:",
        "positive": "ja",
        "negative": "nej",
    },
    "definition": {
        "template": "Hadefuld tale angriber en person eller gruppe baseret på race, religion, køn, seksualitet eller handicap, inkl. trusler og dehumanisering.\nTekst: {text}\nEr teksten hadefuld? Svaret skal indeholde ja eller nej.\nSvar:",
        "positive": "ja",
        "negative": "nej",
    },
    "victim_perspective": {
        "template": "Forestil dig du er den person eller gruppe teksten handler om. Ville du føle dig angrebet eller nedgjort? Givet teksten, er dette hadefuld tale? Svaret skal indeholde ja eller nej.\nTekst: {text}\nSvar:",
        "positive": "ja",
        "negative": "nej",
    },
    "expert_moderator": {
        "template": "Du er en erfaren moderator på sociale medier. Vurder om denne tekst overtræder retningslinjer mod hadefuld tale. Givet teksten, er dette hadefuld tale? Svaret skal indeholde ja eller nej.\nTekst: {text}\nSvar:",
        "positive": "ja",
        "negative": "nej",
    },
    "rules": {
        "template": "Hadefuld tale kræver: (1) angreb på en gruppe baseret på race, religion, køn eller nationalitet, OG (2) trusler, nedværdigende sprog, eller formål om at udelukke/dominere. Identificer målgruppen og efter vurder om teksten opfylder begge krav. Svaret skal indeholde Ja eller Nej.\nTekst: {text}\nSvar:",
        "positive": "ja",
        "negative": "nej",
    },
}

###################################
# FEW SHOT PROMPTING
###################################
# PROMPT_PATTERNS = {
#     "vanilla_qa": {
#         "template": """Givet udsagnet, er dette udsagn hadefuld tale? Angiv venligst dit svar. Svaret skal indeholde Ja eller Nej.

# Eksempel 1:
# Udsagn: Jeg synes debatten om integration er kompleks.
# Svar: Nej

# Eksempel 2:
# Udsagn: Fucking perkere skal ud af landet.
# Svar: Ja

# Eksempel 3:
# Udsagn: Vi bør have en mere nuanceret debat om indvandring.
# Svar: Nej

# Eksempel 4:
# Udsagn: Muslimer er nogle beskidte svin der skal skydes.
# Svar: Ja

# Eksempel 5:
# Udsagn: Det er et svært emne med mange perspektiver.
# Svar: Nej

# Eksempel 6:
# Udsagn: De sorte aber hører ikke til i Danmark.
# Svar: Ja

# Eksempel 7:
# Udsagn: Jeg er uenig i regeringens udlændingepolitik.
# Svar: Nej

# Eksempel 8:
# Udsagn: Homoseksuelle er syge i hovedet og burde aflives.
# Svar: Ja

# Nu din tur:
# Udsagn: {text}
# Svar:""",
#         "positive": "ja",
#         "negative": "nej",
#     },

#     "choice_qa": {
#         "template": """Givet udsagnet, er dette udsagn hadefuld tale? A: Ja. B: Nej. Angiv venligst dit valg (A eller B).

# Eksempel 1:
# Udsagn: Jeg synes debatten om integration er kompleks.
# Svar: B

# Eksempel 2:
# Udsagn: Fucking perkere skal ud af landet.
# Svar: A

# Eksempel 3:
# Udsagn: Vi bør have en mere nuanceret debat om indvandring.
# Svar: B

# Eksempel 4:
# Udsagn: Muslimer er nogle beskidte svin der skal skydes.
# Svar: A

# Eksempel 5:
# Udsagn: Det er et svært emne med mange perspektiver.
# Svar: B

# Eksempel 6:
# Udsagn: De sorte aber hører ikke til i Danmark.
# Svar: A

# Eksempel 7:
# Udsagn: Jeg er uenig i regeringens udlændingepolitik.
# Svar: B

# Eksempel 8:
# Udsagn: Homoseksuelle er syge i hovedet og burde aflives.
# Svar: A

# Nu din tur:
# Udsagn: {text}
# Svar:""",
#         "positive": "a",
#         "negative": "b",
#     },

#     "cloze": {
#         "template": """Givet udsagnet, er dette udsagn hadefuldt eller neutralt? Forudsig venligst [MASK] token, du kan kun svare hadefuldt eller neutralt.

# Eksempel 1:
# Udsagn: Jeg synes debatten om integration er kompleks.
# Svar: Det er et neutralt udsagn.

# Eksempel 2:
# Udsagn: Fucking perkere skal ud af landet.
# Svar: Det er et hadefuldt udsagn.

# Eksempel 3:
# Udsagn: Vi bør have en mere nuanceret debat om indvandring.
# Svar: Det er et neutralt udsagn.

# Eksempel 4:
# Udsagn: Muslimer er nogle beskidte svin der skal skydes.
# Svar: Det er et hadefuldt udsagn.

# Eksempel 5:
# Udsagn: Det er et svært emne med mange perspektiver.
# Svar: Det er et neutralt udsagn.

# Eksempel 6:
# Udsagn: De sorte aber hører ikke til i Danmark.
# Svar: Det er et hadefuldt udsagn.

# Eksempel 7:
# Udsagn: Jeg er uenig i regeringens udlændingepolitik.
# Svar: Det er et neutralt udsagn.

# Eksempel 8:
# Udsagn: Homoseksuelle er syge i hovedet og burde aflives.
# Svar: Det er et hadefuldt udsagn.

# Nu din tur:
# Udsagn: {text}
# Svar: Det er et [MASK] udsagn.""",
#         "positive": "hadefuldt",
#         "negative": "neutralt",
#     },

#     "cot": {
#         "template": """Er dette udsagn hadefuld tale? Dit svar skal kun indeholde Ja eller Nej, og giv derefter din forklaring trin for trin.

# Eksempel 1:
# Udsagn: Jeg synes debatten om integration er kompleks.
# Svar: Nej

# Eksempel 2:
# Udsagn: Fucking perkere skal ud af landet.
# Svar: Ja

# Eksempel 3:
# Udsagn: Vi bør have en mere nuanceret debat om indvandring.
# Svar: Nej

# Eksempel 4:
# Udsagn: Muslimer er nogle beskidte svin der skal skydes.
# Svar: Ja

# Eksempel 5:
# Udsagn: Det er et svært emne med mange perspektiver.
# Svar: Nej

# Eksempel 6:
# Udsagn: De sorte aber hører ikke til i Danmark.
# Svar: Ja

# Eksempel 7:
# Udsagn: Jeg er uenig i regeringens udlændingepolitik.
# Svar: Nej

# Eksempel 8:
# Udsagn: Homoseksuelle er syge i hovedet og burde aflives.
# Svar: Ja

# Nu din tur:
# Udsagn: {text}
# Svar:""",
#         "positive": "ja",
#         "negative": "nej",
#     },

#     "target": {
#         "template": """Givet udsagnet, er dette udsagn hadefuld tale? Dit svar skal kun indeholde Ja eller Nej, og identificer derefter målgruppen eller individet.

# Eksempel 1:
# Udsagn: Jeg synes debatten om integration er kompleks.
# Svar: Nej

# Eksempel 2:
# Udsagn: Fucking perkere skal ud af landet.
# Svar: Ja

# Eksempel 3:
# Udsagn: Vi bør have en mere nuanceret debat om indvandring.
# Svar: Nej

# Eksempel 4:
# Udsagn: Muslimer er nogle beskidte svin der skal skydes.
# Svar: Ja

# Eksempel 5:
# Udsagn: Det er et svært emne med mange perspektiver.
# Svar: Nej

# Eksempel 6:
# Udsagn: De sorte aber hører ikke til i Danmark.
# Svar: Ja

# Eksempel 7:
# Udsagn: Jeg er uenig i regeringens udlændingepolitik.
# Svar: Nej

# Eksempel 8:
# Udsagn: Homoseksuelle er syge i hovedet og burde aflives.
# Svar: Ja

# Nu din tur:
# Udsagn: {text}
# Svar:""",
#         "positive": "ja",
#         "negative": "nej",
#     },

#     # Searle (1976) speech act taxonomy
#     "illocutionary": {
#         "template": """Analyser hvilken sproghandling dette udsagn udfører. Er det en påstand, en opfordring, et løfte, et udtryk for følelser, eller en erklæring? Hadefuld tale fungerer typisk som udtryk for foragt eller opfordring til skade. Givet udsagnet, er dette udsagn hadefuld tale? Svaret skal indeholde ja eller nej.

# Eksempel 1:
# Udsagn: Jeg synes debatten om integration er kompleks.
# Svar: Nej

# Eksempel 2:
# Udsagn: Fucking perkere skal ud af landet.
# Svar: Ja

# Eksempel 3:
# Udsagn: Vi bør have en mere nuanceret debat om indvandring.
# Svar: Nej

# Eksempel 4:
# Udsagn: Muslimer er nogle beskidte svin der skal skydes.
# Svar: Ja

# Eksempel 5:
# Udsagn: Det er et svært emne med mange perspektiver.
# Svar: Nej

# Eksempel 6:
# Udsagn: De sorte aber hører ikke til i Danmark.
# Svar: Ja

# Eksempel 7:
# Udsagn: Jeg er uenig i regeringens udlændingepolitik.
# Svar: Nej

# Eksempel 8:
# Udsagn: Homoseksuelle er syge i hovedet og burde aflives.
# Svar: Ja

# Nu din tur:
# Udsagn: {text}
# Svar:""",
#         "positive": "ja",
#         "negative": "nej",
#     },

#     # Functional linguistics (Halliday)
#     "functional": {
#         "template": """Overvej tekstens sociale funktion: informerer den, underholder, opbygger fællesskab, udelukker andre, eller etablerer dominans? Tekster der udelukker eller dominerer er ofte hadefulde. Givet teksten, er dette hadefuld tale? Svaret skal indeholde ja eller nej.

# Eksempel 1:
# Tekst: Jeg synes debatten om integration er kompleks.
# Svar: Nej

# Eksempel 2:
# Tekst: Fucking perkere skal ud af landet.
# Svar: Ja

# Eksempel 3:
# Tekst: Vi bør have en mere nuanceret debat om indvandring.
# Svar: Nej

# Eksempel 4:
# Tekst: Muslimer er nogle beskidte svin der skal skydes.
# Svar: Ja

# Eksempel 5:
# Tekst: Det er et svært emne med mange perspektiver.
# Svar: Nej

# Eksempel 6:
# Tekst: De sorte aber hører ikke til i Danmark.
# Svar: Ja

# Eksempel 7:
# Tekst: Jeg er uenig i regeringens udlændingepolitik.
# Svar: Nej

# Eksempel 8:
# Tekst: Homoseksuelle er syge i hovedet og burde aflives.
# Svar: Ja

# Nu din tur:
# Tekst: {text}
# Svar:""",
#         "positive": "ja",
#         "negative": "nej",
#     },

#     # Definition-based detection
#     "definition": {
#         "template": """Hadefuld tale angriber en person eller gruppe baseret på race, religion, køn, seksualitet eller handicap, inkl. trusler og dehumanisering.

# Eksempel 1:
# Tekst: Jeg synes debatten om integration er kompleks.
# Svar: Nej

# Eksempel 2:
# Tekst: Fucking perkere skal ud af landet.
# Svar: Ja

# Eksempel 3:
# Tekst: Vi bør have en mere nuanceret debat om indvandring.
# Svar: Nej

# Eksempel 4:
# Tekst: Muslimer er nogle beskidte svin der skal skydes.
# Svar: Ja

# Eksempel 5:
# Tekst: Det er et svært emne med mange perspektiver.
# Svar: Nej

# Eksempel 6:
# Tekst: De sorte aber hører ikke til i Danmark.
# Svar: Ja

# Eksempel 7:
# Tekst: Jeg er uenig i regeringens udlændingepolitik.
# Svar: Nej

# Eksempel 8:
# Tekst: Homoseksuelle er syge i hovedet og burde aflives.
# Svar: Ja

# Nu din tur:
# Tekst: {text}
# Er teksten hadefuld? Svaret skal indeholde ja eller nej.
# Svar:""",
#         "positive": "ja",
#         "negative": "nej",
#     },

#     # Victim/Target Perspective Prompting
#     "victim_perspective": {
#         "template": """Forestil dig du er den person eller gruppe teksten handler om. Ville du føle dig angrebet eller nedgjort? Givet teksten, er dette hadefuld tale? Svaret skal indeholde ja eller nej.

# Eksempel 1:
# Tekst: Jeg synes debatten om integration er kompleks.
# Svar: Nej

# Eksempel 2:
# Tekst: Fucking perkere skal ud af landet.
# Svar: Ja

# Eksempel 3:
# Tekst: Vi bør have en mere nuanceret debat om indvandring.
# Svar: Nej

# Eksempel 4:
# Tekst: Muslimer er nogle beskidte svin der skal skydes.
# Svar: Ja

# Eksempel 5:
# Tekst: Det er et svært emne med mange perspektiver.
# Svar: Nej

# Eksempel 6:
# Tekst: De sorte aber hører ikke til i Danmark.
# Svar: Ja

# Eksempel 7:
# Tekst: Jeg er uenig i regeringens udlændingepolitik.
# Svar: Nej

# Eksempel 8:
# Tekst: Homoseksuelle er syge i hovedet og burde aflives.
# Svar: Ja

# Nu din tur:
# Tekst: {text}
# Svar:""",
#         "positive": "ja",
#         "negative": "nej",
#     },

#     # Expert Role / Social Media Moderator Persona
#     "expert_moderator": {
#         "template": """Du er en erfaren moderator på sociale medier. Vurder om denne tekst overtræder retningslinjer mod hadefuld tale. Givet teksten, er dette hadefuld tale? Svaret skal indeholde ja eller nej.

# Eksempel 1:
# Tekst: Jeg synes debatten om integration er kompleks.
# Svar: Nej

# Eksempel 2:
# Tekst: Fucking perkere skal ud af landet.
# Svar: Ja

# Eksempel 3:
# Tekst: Vi bør have en mere nuanceret debat om indvandring.
# Svar: Nej

# Eksempel 4:
# Tekst: Muslimer er nogle beskidte svin der skal skydes.
# Svar: Ja

# Eksempel 5:
# Tekst: Det er et svært emne med mange perspektiver.
# Svar: Nej

# Eksempel 6:
# Tekst: De sorte aber hører ikke til i Danmark.
# Svar: Ja

# Eksempel 7:
# Tekst: Jeg er uenig i regeringens udlændingepolitik.
# Svar: Nej

# Eksempel 8:
# Tekst: Homoseksuelle er syge i hovedet og burde aflives.
# Svar: Ja

# Nu din tur:
# Tekst: {text}
# Svar:""",
#         "positive": "ja",
#         "negative": "nej",
#     },

#     "rules": {
#         "template": """Hadefuld tale kræver: (1) angreb på en gruppe baseret på race, religion, køn eller nationalitet, OG (2) trusler, nedværdigende sprog, eller formål om at udelukke/dominere.

# Eksempel 1:
# Tekst: Jeg synes debatten om integration er kompleks.
# Svar: Nej

# Eksempel 2:
# Tekst: Fucking perkere skal ud af landet.
# Svar: Ja

# Eksempel 3:
# Tekst: Vi bør have en mere nuanceret debat om indvandring.
# Svar: Nej

# Eksempel 4:
# Tekst: Muslimer er nogle beskidte svin der skal skydes.
# Svar: Ja

# Eksempel 5:
# Tekst: Det er et svært emne med mange perspektiver.
# Svar: Nej

# Eksempel 6:
# Tekst: De sorte aber hører ikke til i Danmark.
# Svar: Ja

# Eksempel 7:
# Tekst: Jeg er uenig i regeringens udlændingepolitik.
# Svar: Nej

# Eksempel 8:
# Tekst: Homoseksuelle er syge i hovedet og burde aflives.
# Svar: Ja

# Nu din tur:
# Tekst: {text}
# Opfylder teksten begge krav? Svaret skal indeholde Ja eller Nej.
# Svar:""",
#         "positive": "ja",
#         "negative": "nej",
#     },
# }

# Noise-aware prompts (commented out)

# NOISE_PREFIX_DA = """Du l\u00e6ser r\u00e5 tekst fra sociale medier. Teksten kan indeholde forskellige former for st\u00f8j s\u00e5som:
# - Slang, forkortelser og stavefejl
# - URLs og links
# - Mentions (@brugernavn)
# - Hashtags (#emne)
# - Emojis og specielle tegn
# - Gentagne tegn og uformelt sprog

# Se forbi denne st\u00f8j og vurder det faktiske budskab.

# """

# PROMPT_PATTERNS = {
#     "vanilla_qa": {
#         "template": NOISE_PREFIX_DA + "Givet udsagnet, er dette udsagn hadefuld tale? Angiv venligst dit svar. Svaret skal indeholde Ja eller Nej.\nUdsagn: {text}\nSvar:",
#         "positive": "ja", - Cross Pattern
#         "negative": "nej",
#     },
#     "choice_qa": {
#         "template": NOISE_PREFIX_DA + "Givet udsagnet, er dette udsagn hadefuld tale? A: Ja. B: Nej. Angiv venligst dit valg (A eller B).\nUdsagn: {text}\nSvar:",
#         "positive": "a",
#         "negative": "b",
#     },
#     "cloze": {
#         "template": NOISE_PREFIX_DA + "Givet udsagnet, er dette udsagn hadefuldt eller neutralt? Forudsig venligst [MASK] token, du kan kun svare hadefuldt eller neutralt.\nUdsagn: {text}\nSvar: Det er et [MASK] udsagn.",
#         "positive": "hadefuldt",
#         "negative": "neutralt",
#     },
#     "cot": {
#         "template": NOISE_PREFIX_DA + "Er dette udsagn hadefuld tale? Dit svar skal kun indeholde Ja eller Nej, og giv derefter din forklaring trin for trin.\nUdsagn: {text}\nSvar:",
#         "positive": "ja",
#         "negative": "nej",
#     },
#     "target": {
#         "template": NOISE_PREFIX_DA + "Givet udsagnet, er dette udsagn hadefuld tale? Dit svar skal kun indeholde Ja eller Nej, og identificer derefter m\u00e5lgruppen eller individet.\nUdsagn: {text}\nSvar:",
#         "positive": "ja",
#         "negative": "nej",
#     },
#     "illocutionary": {
#         "template": NOISE_PREFIX_DA + "Analyser hvilken sproghandling dette udsagn udf\u00f8rer. Er det en p\u00e5stand, en opfordring, et l\u00f8fte, et udtryk for f\u00f8lelser, eller en erkl\u00e6ring? Hadefuld tale fungerer typisk som udtryk for foragt eller opfordring til skade. Givet udsagnet, er dette udsagn hadefuld tale? Svaret skal indeholde ja eller nej.\nUdsagn: {text}\nSvar:",
#         "positive": "ja",
#         "negative": "nej",
#     },
#     "functional": {
#         "template": NOISE_PREFIX_DA + "Overvej tekstens sociale funktion: informerer den, underholder, opbygger f\u00e6llesskab, udelukker andre, eller etablerer dominans? Tekster der udelukker eller dominerer er ofte hadefulde. Givet teksten, er dette hadefuld tale? Svaret skal indeholde ja eller nej.\nTekst: {text}\nSvar:",
#         "positive": "ja",
#         "negative": "nej",
#     },
#     "definition": {
#         "template": NOISE_PREFIX_DA + "Hadefuld tale angriber en person eller gruppe baseret p\u00e5 race, religion, k\u00f8n, seksualitet eller handicap, inkl. trusler og dehumanisering.\nTekst: {text}\nEr teksten hadefuld? Svaret skal indeholde ja eller nej.\nSvar:",
#         "positive": "ja",
#         "negative": "nej",
#     },
#     "victim_perspective": {
#         "template": NOISE_PREFIX_DA + "Forestil dig du er den person eller gruppe teksten handler om. Ville du f\u00f8le dig angrebet eller nedgjort? Givet teksten, er dette hadefuld tale eller nej? Svaret skal indeholde ja eller nej.\nTekst: {text}\nSvar:",
#         "positive": "ja",
#         "negative": "nej",
#     },
#     "expert_moderator": {
#         "template": NOISE_PREFIX_DA + "Du er en erfaren moderator p\u00e5 sociale medier. Vurder om denne tekst overtr\u00e6der retningslinjer mod hadefuld tale eller nej. Givet teksten, er dette hadefuld tale? Svaret skal indeholde ja eller nej.\nTekst: {text}\nSvar:",
#         "positive": "ja",
#         "negative": "nej",
#     },
#         "rules": {
#         "template": NOISE_PREFIX_DA + "Hadefuld tale kr\u00e6ver: (1) angreb p\u00e5 en gruppe baseret p\u00e5 race, religion, k\u00f8n eller nationalitet, OG (2) trusler, nedv\u00e6rdigende sprog, eller form\u00e5l om at udelukke eller dominere. Identificer m\u00e5lgruppen og efter vurder om teksten opfylder begge krav. Svaret skal indeholde Ja eller Nej.\nTekst: {text}\nSvar:",
#         "positive": "ja",
#         "negative": "nej",
#     },
# }

# ASCII encoding prompt version
# PROMPT_PATTERNS = {
#     "vanilla_qa": {
#         "template": "Givet udsagnet, er dette udsagn hadefuld tale? Angiv venligst dit svar. Svaret skal indeholde Ja eller Nej.\nUdsagn: {text}\nSvar:",
#         "positive": "ja",
#         "negative": "nej",
#     },
#     "choice_qa": {
#         "template": "Givet udsagnet, er dette udsagn hadefuld tale? A: Ja. B: Nej. Angiv venligst dit valg (A eller B).\nUdsagn: {text}\nSvar:",
#         "positive": "a",
#         "negative": "b",
#     },
#     "cloze": {
#         "template": "Givet udsagnet, er dette udsagn hadefuldt eller neutralt? Forudsig venligst [MASK] token, du kan kun svare hadefuldt eller neutralt.\nUdsagn: {text}\nSvar: Det er et [MASK] udsagn.",
#         "positive": "hadefuldt",
#         "negative": "neutralt",
#     },
#     "cot": {
#         "template": "Er dette udsagn hadefuld tale? Dit svar skal kun indeholde Ja eller Nej, og giv derefter din forklaring trin for trin.\nUdsagn: {text}\nSvar:",
#         "positive": "ja",
#         "negative": "nej",
#     },
#     "target": {
#         "template": "Givet udsagnet, er dette udsagn hadefuld tale? Dit svar skal kun indeholde Ja eller Nej, og identificer derefter den gruppe eller person som er maal for udsagnet.\nUdsagn: {text}\nSvar:",
#         "positive": "ja",
#         "negative": "nej",
#     },
#
#     "illocutionary": {
#         # Speech Act Theory framework - analyzes communicative intent
#         # Reference: Austin (1962) "How to Do Things with Words"; Searle (1969) Speech Acts
#         # Theoretical basis: Hate speech performs specific illocutionary acts (expressing contempt, inciting harm)
#         # Note: The second sentence is DEFINITIONAL (describes what hate speech does), not leading
#         "template": "Analyser hvilken sproghandling dette udsagn udfoerer. Er det en paastand, en opfordring, et loefte, et udtryk for foelelser, eller en erklaering? Hadefuld tale fungerer typisk som udtryk for foragt eller opfordring til skade. Givet udsagnet, er dette udsagn hadefuld tale? Svaret skal indeholde ja eller nej.\nUdsagn: {text}\nSvar:",
#         "positive": "ja",
#         "negative": "nej",
#     },
#
#     "functional": {
#         # Functional Linguistics framework - analyzes social function of text
#         # Reference: Halliday (1985) Systemic Functional Linguistics; Roettger et al. (2021) HateCheck
#         # UNBIASED VERSION: Removed "Tekster der udelukker eller dominerer er ofte hadefulde"
#         # Reason: That phrase was LEADING - directly linked analytical category to classification
#         # (CEA-LIST CheckThat! 2025; Learn Prompting Debiasing Guide)
#         "template": "Overvej tekstens sociale funktion: informerer den, underholder, opbygger faellesskab, udelukker andre, eller etablerer dominans? Givet teksten, er dette hadefuld tale? Svaret skal indeholde ja eller nej.\nTekst: {text}\nSvar:",
#         "positive": "ja",
#         "negative": "nej",
#     },
#
#     "definition": {
#         # Explicit definition-based classification
#         # Reference: Poletto et al. (2021) "Resources and benchmark corpora for hate speech detection"
#         # Reference: de Gibert et al. (2018) Stormfront dataset annotation guidelines
#         # Note: "inkl. trusler og dehumanisering" is INFORMATIVE (clarifies what "attack" means)
#         # This aligns with Ross et al. (2017): "clearer definition is necessary for reliable classification"
#         "template": "Hadefuld tale angriber en person eller gruppe baseret paa race, religion, koen, seksualitet eller handicap, inkl. trusler og dehumanisering.\nTekst: {text}\nEr teksten hadefuld? Svaret skal indeholde ja eller nej.\nSvar:",
#         "positive": "ja",
#         "negative": "nej",
#     },
#
#     "victim_perspective": {
#         # Perspective-taking / empathy-based framework
#         # Reference: Chiril et al. (2022) "Emotionally Informed Hate Speech Detection: A Multi-target Perspective"
#         # UNBIASED VERSION: Changed "Ville du foele dig angrebet eller nedgjort?" to neutral framing
#         # Reason: Original only offered NEGATIVE feeling options, priming toward positive classification
#         # Fix based on: Assimakopoulos et al. (2020) - "objective criteria rather than subjective categories"
#         "template": "Forestil dig du er den person eller gruppe teksten handler om. Hvordan ville du opfatte denne tekst? Givet teksten, er dette hadefuld tale? Svaret skal indeholde ja eller nej.\nTekst: {text}\nSvar:",
#         "positive": "ja",
#         "negative": "nej",
#     },
#
#     "expert_moderator": {
#         # Persona/role-based prompting framework
#         # Reference: White et al. (2023) "A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT"
#         # Note: Persona is NEUTRAL - does not suggest what answer should be
#         # Caveat: Research shows personas may not improve classification accuracy
#         # (PromptHub 2024; "When 'A Helpful Assistant' Is Not Really Helpful" 2024)
#         # but they do not introduce DIRECTIONAL bias in this case
#         "template": "Du er en erfaren moderator paa sociale medier. Vurder om denne tekst overtraeder retningslinjer mod hadefuld tale. Givet teksten, er dette hadefuld tale? Svaret skal indeholde ja eller nej.\nTekst: {text}\nSvar:",
#         "positive": "ja",
#         "negative": "nej",
#     },
#     "rules": {
#         "template": "Hadefuld tale kraever: (1) angreb paa en gruppe baseret paa race, religion, koennet eller nationalitet, OG (2) trusler, nedvaerdigende sprog, eller formaal om at udelukke/dominere. Identificer maalgruppen og efter vurder om teksten opfylder begge krav. Svaret skal indeholde Ja eller Nej.\nTekst: {text}\nSvar:",
#         "positive": "ja",
#         "negative": "nej",
#     },
#
# }

VERBAL_PROMPT = (
    "Givet udsagnet, er dette udsagn hadefuld tale? "
    "Svar med 'Ja' eller 'Nej' og din tillid fra 0% til 100%.\n"
    "Udsagn: {text}\nSvar:"
)

# Visualization settings
MODEL_COLORS = {
    "llama": "#2E7D32",  # Green
    "mistral": "#1565C0",  # Blue
    "gemma": "#E65100",  # Orange
    "qwen": "#B71C1C",  # Red
}
