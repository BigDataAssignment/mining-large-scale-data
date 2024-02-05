import numpy as np
import spacy
import nltk
from nltk.corpus import words as nltk_words

nlp = spacy.load("en_core_web_sm")
nltk.download("words")
nltk.download('vader_lexicon')
english_words = set(nltk_words.words())

# borrowed from the research: https://github.com/zhukovanadezhda/psy-ner/tree/main
# Following paper: https://www.researchgate.net/publication/358779855_Deep_Learning-based_Detection_of_Psychiatric_Attributes_from_German_Mental_Health_Records
# see: https://spacy.io/usage/processing-pipelines
psy_ner = spacy.load("./model/psy_ner")
from nltk.sentiment import SentimentIntensityAnalyzer
from empath import Empath

lexicon = Empath()
sia = SentimentIntensityAnalyzer()

MH_NER = [
    "ANXIETY DISORDERS",
    "BIPOLAR DISORDERS",
    "DEPRESSIVE DISORDERS",
    "DISRUPTIVE IMPULSE-CONTROL, AND CONDUCT DISORDERS",
    "DISSOCIATIVE DISORDERS",
    "EATING DISORDERS",
    "NEURO-COGNITIVE DISORDERS",
    "NEURO-DEVELOPMENTAL DISORDERS",
    "OBSESSIVE-COMPULSIVE AND RELATED DISORDERS",
    "PERSONALITY DISORDERS",
    "PSYCHEDELIC DRUGS",
    "SCHIZOPHRENIA SPECTRUM AND OTHER PSYCHOTIC DISORDERS",
    "SEXUAL DYSFUNCTIONS",
    "SLEEP-WAKE DISORDERS",
    "SOMATIC SYMPTOM RELATED DISORDERS",
    "SUBSTANCE-RELATED DISORDERS",
    "SYMPTOMS",
    "TRAUMA AND STRESS RELATED DISORDERS",
]


def create_rel_feature(row):
    def _extract_relations(text):
        relations = []
        if not isinstance(text, str):
            return relations
        doc = nlp(text)

        for sent in doc.sents:
            for token in sent:
                if token.dep_ in ["nsubj", "dobj"]:
                    relations.append((token.head.text, token.dep_, token.text))
        return relations

    title_relations = _extract_relations(row["title"])
    text_relations = _extract_relations(row["selftext"])
    all_relations = title_relations + text_relations
    return all_relations


def create_psylabel_feature(row):
    def _extract_psy_labels(text):
        mh_labels = {}
        if not isinstance(text, str):
            return mh_labels

        doc = psy_ner(text)
        for ent in doc.ents:
            if ent.label_ in MH_NER:
                if ent.label_ not in mh_labels:
                    mh_labels[ent.label_] = set()
                mh_labels[ent.label_].add(ent.text)
        for label in mh_labels:
            mh_labels[label] = list(mh_labels[label])

        return mh_labels

    combined_text = row["title"] + " " + row["selftext"]
    combined_labels = _extract_psy_labels(combined_text)
    return combined_labels


# See: https://github.com/Ejhfast/empath-client/tree/master
EMPATH_CATS = [
    "help",
    "violence",
    "sleep",
    "medical_emergency",
    "cold",
    "hate",
    "cheerfulness",
    "aggression",
    "envy",
    "anticipation",
    "health",
    "pride",
    "nervousness",
    "weakness",
    "horror",
    "swearing_terms",
    "suffering",
    "sexual",
    "fear",
    "monster",
    "irritability",
    "exasperation",
    "ridicule",
    "neglect",
    "fight",
    "dominant_personality",
    "injury",
    "rage",
    "science",
    "work",
    "optimism",
    "warmth",
    "sadness",
    "emotional",
    "joy",
    "shame",
    "torment",
    "anger",
    "strength",
    "ugliness",
    "pain",
    "negative_emotion",
    "positive_emotion",
]

def create_sentiment_feature(row):
    def _get_vader_sentiment(text):
        score = sia.polarity_scores(text)
        return score["compound"] if score is not None else np.NaN

    combined_text = row["title"] + " " + row["selftext"]
    combined_labels = _get_vader_sentiment(combined_text)
    return combined_labels


def create_emotional_categories_scores_feature(row):
    def _get_empath_sentiment(text):
        scores = lexicon.analyze(text, categories=EMPATH_CATS, normalize=True)
        if scores is not None:
            return {category: round(score, 2) for category, score in scores.items()}
        else:
            return {}  # Return an empty dictionary if scores is None

    combined_text = row["title"] + " " + row["selftext"]
    combined_labels = _get_empath_sentiment(combined_text)
    return combined_labels