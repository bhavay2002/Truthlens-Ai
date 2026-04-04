"""
Central emotion schema for TruthLens.

This module defines the canonical emotion labels and lexicon used
throughout the system to ensure consistency across feature extraction,
analysis modules, and model training.
"""

# -----------------------------------------------------
# Emotion Labels (TruthLens Standard)
# -----------------------------------------------------

EMOTION_LABELS = [
    "neutral",
    "admiration",
    "approval",
    "gratitude",
    "annoyance",
    "amusement",
    "curiosity",
    "disapproval",
    "love",
    "optimism",
    "anger",
    "joy",
    "confusion",
    "sadness",
    "disappointment",
    "realization",
    "caring",
    "surprise",
    "excitement",
    "disgust",
]

# -----------------------------------------------------
# Emotion Lexicon
# -----------------------------------------------------

EMOTION_TERMS = {

    "admiration": {
        "admire","admiration","respect","praise","commend","applaud",
        "appreciate","revere","esteem","honor","look_up_to","inspire"
    },

    "approval": {
        "approve","approval","support","endorse","accept","agree",
        "back","validate","favor","ratify","sanction"
    },

    "gratitude": {
        "thanks","thank","thankful","grateful","gratitude",
        "appreciation","indebted","obliged","much_obliged"
    },

    "annoyance": {
        "annoy","annoying","irritate","irritating","bother",
        "frustrate","frustrating","aggravate","aggravating",
        "disturb","disturbing"
    },

    "amusement": {
        "funny","amusing","hilarious","laugh","laughing",
        "entertaining","comic","comical","witty","playful"
    },

    "curiosity": {
        "curious","curiosity","wonder","wondering","intrigued",
        "interested","interest","inquisitive","question",
        "explore","exploration"
    },

    "disapproval": {
        "disapprove","disapproval","criticize","criticism",
        "condemn","condemnation","reject","denounce",
        "oppose","objection"
    },

    "love": {
        "love","adore","adoration","affection","fond",
        "fondness","cherish","devotion","passion","care_deeply"
    },

    "optimism": {
        "hope","hopeful","optimistic","optimism","positive",
        "encouraging","promising","confidence","confident",
        "bright_future"
    },

    "anger": {
        "anger","angry","furious","rage","outrage","fury",
        "irate","resent","resentment","enraged","hostile"
    },

    "joy": {
        "joy","joyful","happy","happiness","delighted",
        "delight","pleased","glad","cheerful","elated"
    },

    "confusion": {
        "confused","confusion","uncertain","uncertainty",
        "puzzled","perplexed","unclear","misunderstand",
        "ambiguous","bewildered"
    },

    "sadness": {
        "sad","sadness","depressed","depression","unhappy",
        "sorrow","sorrowful","gloomy","melancholy","grief"
    },

    "disappointment": {
        "disappointed","disappointment","letdown","dismayed",
        "discouraged","regret","regretful","frustrated_expectations"
    },

    "realization": {
        "realize","realization","realise","understand",
        "recognize","recognise","awareness","discover",
        "figure_out"
    },

    "caring": {
        "care","caring","concern","concerned","compassion",
        "empathetic","empathy","supportive","kindness"
    },

    "surprise": {
        "surprise","surprised","astonished","astonishment",
        "shocked","shock","unexpected","startled","amazed"
    },

    "excitement": {
        "excited","exciting","thrilled","thrill",
        "enthusiastic","enthusiasm","eager","anticipation"
    },

    "disgust": {
        "disgust","disgusting","gross","repulsive",
        "revolting","nauseating","sickening","abhorrent"
    },
}