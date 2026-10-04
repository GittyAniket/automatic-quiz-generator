from flask import Flask, render_template, request
import nltk
import re
import random

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================================================
# NLTK SETUP
# =========================================================

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

app = Flask(__name__)

app.secret_key = "automatic-quiz-generator"

stop_words = set(stopwords.words("english"))


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):
    """Clean extra spaces and line breaks."""

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================================================
# NORMALIZE CONCEPTS
# =========================================================

def normalize_concept(concept):
    """
    Normalize concept names so that:
    AI and Artificial Intelligence are treated as the same concept.
    """

    concept = concept.strip().lower()

    concept = re.sub(r"[^\w\s-]", "", concept)

    aliases = {
        "ai": "artificial intelligence",
        "artificial intelligence": "artificial intelligence",
        "ml": "machine learning",
        "machine learning": "machine learning",
        "nlp": "natural language processing",
        "natural language processing": "natural language processing",
        "cv": "computer vision",
        "computer vision": "computer vision",
        "dl": "deep learning",
        "deep learning": "deep learning"
    }

    return aliases.get(concept, concept)


# =========================================================
# DISPLAY NAME
# =========================================================

def display_concept(concept):
    """
    Convert normalized concepts into clean display names.
    """

    normalized = normalize_concept(concept)

    display_names = {
        "artificial intelligence": "Artificial Intelligence",
        "machine learning": "Machine Learning",
        "natural language processing": "Natural Language Processing",
        "computer vision": "Computer Vision",
        "deep learning": "Deep Learning"
    }

    return display_names.get(
        normalized,
        concept.strip().title()
    )


# =========================================================
# TF-IDF KEYWORD EXTRACTION
# =========================================================

def extract_keywords(text, num_keywords=30):
    """
    Extract important words and short phrases using TF-IDF.
    This is used as a supporting NLP technique.
    """

    text = clean_text(text)

    if not text:
        return []

    sentences = sent_tokenize(text)

    if not sentences:
        return []

    try:

        # TF-IDF requires at least 2 documents to calculate
        # meaningful document-level differences.
        # We therefore use sentences as documents.

        if len(sentences) == 1:

            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=100
            )

            matrix = vectorizer.fit_transform([text])

        else:

            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=100
            )

            matrix = vectorizer.fit_transform(sentences)


        scores = matrix.sum(axis=0).A1

        terms = vectorizer.get_feature_names_out()

        ranked_terms = sorted(
            zip(terms, scores),
            key=lambda x: x[1],
            reverse=True
        )

        keywords = []

        for term, score in ranked_terms:

            term = term.strip()

            if len(term) < 3:
                continue

            if term.lower() in stop_words:
                continue

            keywords.append(term)

            if len(keywords) >= num_keywords:
                break

        return keywords

    except Exception:
        return []


# =========================================================
# CONCEPT EXTRACTION
# =========================================================

def extract_concepts(text):
    """
    Detect important concepts/topics from informative sentences.

    Examples:

    Artificial Intelligence is...
        -> Artificial Intelligence

    Machine learning allows...
        -> Machine learning

    Natural language processing is...
        -> Natural language processing
    """

    sentences = sent_tokenize(
        clean_text(text)
    )

    concepts = []

    generic_concepts = {
        "this",
        "that",
        "it",
        "they",
        "he",
        "she",
        "these",
        "those"
    }

    # -----------------------------------------------------
    # Definition pattern
    # -----------------------------------------------------

    definition_pattern = re.compile(
        r"^([A-Z][A-Za-z0-9-]*(?:\s+[A-Za-z0-9-]+){0,5})"
        r"\s+(?:is|are|was|were)\b"
    )

    # -----------------------------------------------------
    # Function pattern
    # -----------------------------------------------------

    function_pattern = re.compile(
        r"^([A-Z][A-Za-z0-9-]*(?:\s+[A-Za-z0-9-]+){0,5})"
        r"\s+(?:allows|enables|helps|provides|supports)\b",
        re.IGNORECASE
    )

    for sentence in sentences:

        sentence = sentence.strip()

        concept = None

        # Try definition pattern first
        match = definition_pattern.search(sentence)

        if match:
            concept = match.group(1).strip()

        else:
            # Try function pattern
            match = function_pattern.search(sentence)

            if match:
                concept = match.group(1).strip()


        if not concept:
            continue


        concept = re.sub(
            r"[.,:;!?]+$",
            "",
            concept
        )

        if concept.lower() in generic_concepts:
            continue

        if not 1 <= len(concept.split()) <= 6:
            continue


        normalized = normalize_concept(
            concept
        )


        # Remove duplicate concepts
        if normalized not in [
            normalize_concept(x)
            for x in concepts
        ]:

            concepts.append(concept)


    return concepts


# =========================================================
# GET MEANINGFUL CONCEPTS
# =========================================================

def get_candidate_concepts(text):
    """
    Return meaningful concepts for use as:
    - correct answers
    - MCQ distractors

    This is intentionally concept-focused instead of
    taking random words from the article.
    """

    concepts = extract_concepts(text)

    clean_concepts = []

    for concept in concepts:

        normalized = normalize_concept(
            concept
        )

        # Ignore generic single words
        if normalized in {
            "computer",
            "technology",
            "science",
            "data",
            "system",
            "method",
            "field",
            "branch"
        }:
            continue


        # Avoid duplicate concepts
        if normalized not in [
            normalize_concept(x)
            for x in clean_concepts
        ]:

            clean_concepts.append(
                display_concept(concept)
            )


    return clean_concepts


# =========================================================
# QUESTION GENERATION
# =========================================================

def generate_questions(text, num_questions=5):

    text = clean_text(text)

    sentences = sent_tokenize(text)

    questions = []


    for sentence in sentences:

        sentence = sentence.strip()

        if len(sentence.split()) < 7:
            continue


        # =================================================
        # PATTERN 1
        # "AI is widely used in healthcare..."
        # =================================================

        match = re.match(
            r"^([A-Z][A-Za-z0-9-]*(?:\s+[A-Za-z0-9-]+){0,5})"
            r"\s+(?:is|are|was|were)\s+"
            r"widely used in\s+(.+)$",
            sentence,
            re.IGNORECASE
        )


        if match:

            subject = match.group(1).strip()

            if 1 <= len(subject.split()) <= 6:

                questions.append({

                    "question":
                        f"Where is {display_concept(subject)} "
                        f"widely used?",

                    "answer":
                        display_concept(subject)

                })

                continue


        # =================================================
        # PATTERN 2
        # "X is a/an Y..."
        # =================================================

        match = re.match(
            r"^([A-Z][A-Za-z0-9-]*(?:\s+[A-Za-z0-9-]+){0,5})"
            r"\s+(?:is|are)\s+"
            r"(?:a|an)\s+(.+)$",
            sentence
        )


        if match:

            subject = match.group(1).strip()

            if 1 <= len(subject.split()) <= 6:

                questions.append({

                    "question":
                        f"What is {display_concept(subject)}?",

                    "answer":
                        display_concept(subject)

                })

                continue


        # =================================================
        # PATTERN 3
        # "X is..."
        # =================================================

        match = re.match(
            r"^([A-Z][A-Za-z0-9-]*(?:\s+[A-Za-z0-9-]+){0,5})"
            r"\s+(?:is|are|was|were)\s+(.+)$",
            sentence
        )


        if match:

            subject = match.group(1).strip()

            if 1 <= len(subject.split()) <= 6:

                questions.append({

                    "question":
                        f"What is {display_concept(subject)}?",

                    "answer":
                        display_concept(subject)

                })

                continue


        # =================================================
        # PATTERN 4
        # "X allows..."
        # =================================================

        match = re.match(
            r"^([A-Z][A-Za-z0-9-]*(?:\s+[A-Za-z0-9-]+){0,5})"
            r"\s+(allows|enables|helps)\s+(.+)$",
            sentence,
            re.IGNORECASE
        )


        if match:

            subject = match.group(1).strip()

            verb = match.group(2).lower()

            if 1 <= len(subject.split()) <= 6:

                subject_display = display_concept(
                    subject
                )

                if verb == "allows":

                    question_text = (
                        f"What does "
                        f"{subject_display} allow?"
                    )

                elif verb == "enables":

                    question_text = (
                        f"What does "
                        f"{subject_display} enable?"
                    )

                else:

                    question_text = (
                        f"What does "
                        f"{subject_display} help with?"
                    )


                questions.append({

                    "question":
                        question_text,

                    "answer":
                        subject_display

                })

                continue


        # =================================================
        # PATTERN 5
        # "X focuses on..."
        # =================================================

        match = re.match(
            r"^([A-Z][A-Za-z0-9-]*(?:\s+[A-Za-z0-9-]+){0,5})"
            r"\s+focuses on\s+(.+)$",
            sentence,
            re.IGNORECASE
        )


        if match:

            subject = match.group(1).strip()

            if 1 <= len(subject.split()) <= 6:

                questions.append({

                    "question":
                        f"What is the main focus of "
                        f"{display_concept(subject)}?",

                    "answer":
                        display_concept(subject)

                })

                continue


        # =================================================
        # PATTERN 6
        # "X refers to..."
        # =================================================

        match = re.match(
            r"^([A-Z][A-Za-z0-9-]*(?:\s+[A-Za-z0-9-]+){0,5})"
            r"\s+refers to\s+(.+)$",
            sentence,
            re.IGNORECASE
        )


        if match:

            subject = match.group(1).strip()

            if 1 <= len(subject.split()) <= 6:

                questions.append({

                    "question":
                        f"What does "
                        f"{display_concept(subject)} "
                        f"refer to?",

                    "answer":
                        display_concept(subject)

                })

                continue


        # =================================================
        # FALLBACK
        # =================================================

        keywords = extract_keywords(
            sentence,
            10
        )


        if keywords:

            answer = keywords[0]

            pattern = (
                r"\b" +
                re.escape(answer) +
                r"\b"
            )


            replaced = re.sub(
                pattern,
                "_____",
                sentence,
                count=1,
                flags=re.IGNORECASE
            )


            if replaced != sentence:

                questions.append({

                    "question":
                        "Complete the statement: "
                        + replaced,

                    "answer":
                        answer

                })


    # =====================================================
    # REMOVE DUPLICATES
    # =====================================================

    unique_questions = []

    seen = set()

    for q in questions:

        question_key = q[
            "question"
        ].lower()

        if question_key not in seen:

            unique_questions.append(q)

            seen.add(question_key)


    # Shuffle questions
    random.shuffle(
        unique_questions
    )


    return unique_questions[
        :num_questions
    ]


# =========================================================
# MCQ GENERATION
# =========================================================

def generate_mcqs(text, num_questions=5):

    questions = generate_questions(
        text,
        num_questions
    )


    # Get meaningful concepts
    concepts = get_candidate_concepts(
        text
    )


    mcqs = []


    for q in questions:

        correct_answer = q[
            "answer"
        ]


        correct_normalized = normalize_concept(
            correct_answer
        )


        # =================================================
        # FIND CONCEPT-BASED DISTRACTORS
        # =================================================

        distractors = []


        for concept in concepts:

            normalized = normalize_concept(
                concept
            )


            # Don't use the correct answer
            if normalized == correct_normalized:
                continue


            # Don't use equivalent aliases
            if (
                normalized in correct_normalized
                or
                correct_normalized in normalized
            ):
                continue


            # Don't duplicate
            if concept.lower() in [
                x.lower()
                for x in distractors
            ]:
                continue


            distractors.append(
                concept
            )


        # Shuffle concept distractors
        random.shuffle(
            distractors
        )


        # =================================================
        # ADD MORE CONCEPTS IF NECESSARY
        # =================================================

        if len(distractors) < 3:

            extra_concepts = extract_concepts(
                text
            )


            for concept in extra_concepts:

                display_name = display_concept(
                    concept
                )

                normalized = normalize_concept(
                    display_name
                )


                if normalized == correct_normalized:
                    continue


                if (
                    normalized in correct_normalized
                    or
                    correct_normalized in normalized
                ):
                    continue


                if display_name.lower() in [
                    x.lower()
                    for x in distractors
                ]:
                    continue


                distractors.append(
                    display_name
                )


                if len(distractors) >= 3:
                    break


        # =================================================
        # LAST RESORT: CLEAN KEYWORDS
        # =================================================

        if len(distractors) < 3:

            keywords = extract_keywords(
                text,
                50
            )


            ignored_keywords = {
                "using",
                "used",
                "allows",
                "allow",
                "enables",
                "enable",
                "helps",
                "help",
                "provides",
                "supports",
                "branch",
                "important",
                "part",
                "computer",
                "computers",
                "technology",
                "field",
                "method",
                "data",
                "system",
                "intelligent",
                "machines",
                "machine"
            }


            for keyword in keywords:

                keyword_clean = keyword.strip()

                normalized = normalize_concept(
                    keyword_clean
                )


                if len(keyword_clean) < 4:
                    continue


                if normalized == correct_normalized:
                    continue


                if keyword_clean.lower() in ignored_keywords:
                    continue


                if (
                    normalized in correct_normalized
                    or
                    correct_normalized in normalized
                ):
                    continue


                if keyword_clean.lower() in [
                    x.lower()
                    for x in distractors
                ]:
                    continue


                distractors.append(
                    keyword_clean.title()
                )


                if len(distractors) >= 3:
                    break


        # =================================================
        # NEED EXACTLY 3 WRONG OPTIONS
        # =================================================

        if len(distractors) < 3:

            continue


        distractors = distractors[:3]


        # =================================================
        # CREATE FOUR OPTIONS
        # =================================================

        options = (
            distractors
            + [correct_answer]
        )


        # Shuffle answer position
        random.shuffle(
            options
        )


        mcqs.append({

            "question":
                q["question"],

            "options":
                options,

            "answer":
                correct_answer

        })


    return mcqs


# =========================================================
# HOME ROUTE
# =========================================================

@app.route("/", methods=["GET", "POST"])
def home():

    quiz = []

    text = ""

    message = ""

    result = None


    # =====================================================
    # POST
    # =====================================================

    if request.method == "POST":

        action = request.form.get(
            "action"
        )


        # =================================================
        # GENERATE QUIZ
        # =================================================

        if action == "generate":

            text = request.form.get(
                "text",
                ""
            )


            try:

                num_questions = int(
                    request.form.get(
                        "num_questions",
                        5
                    )
                )

            except Exception:

                num_questions = 5


            # Keep questions between 1 and 10
            num_questions = max(
                1,
                min(
                    num_questions,
                    10
                )
            )


            if text.strip():

                quiz = generate_mcqs(
                    text,
                    num_questions
                )


                if not quiz:

                    message = (
                        "Not enough suitable concepts "
                        "were found. Please enter a "
                        "longer article containing "
                        "multiple concepts."
                    )


            else:

                message = (
                    "Please enter some text."
                )


        # =================================================
        # SUBMIT QUIZ
        # =================================================

        elif action == "submit":

            try:

                question_count = int(
                    request.form.get(
                        "question_count",
                        0
                    )
                )

            except Exception:

                question_count = 0


            # ---------------------------------------------
            # Read submitted questions
            # ---------------------------------------------

            for i in range(
                question_count
            ):

                question = request.form.get(
                    f"question_{i}",
                    ""
                )


                correct_answer = request.form.get(
                    f"correct_{i}",
                    ""
                )


                selected_answer = request.form.get(
                    f"answer_{i}"
                )


                quiz.append({

                    "question":
                        question,

                    "answer":
                        correct_answer,

                    "selected":
                        selected_answer

                })


            # ---------------------------------------------
            # Calculate score
            # ---------------------------------------------

            score = 0


            for q in quiz:

                selected = q[
                    "selected"
                ]

                correct = q[
                    "answer"
                ]


                if (
                    selected
                    and
                    selected.strip().lower()
                    ==
                    correct.strip().lower()
                ):

                    score += 1


            total = len(quiz)


            if total > 0:

                percentage = round(
                    (score / total) * 100
                )

            else:

                percentage = 0


            result = {

                "score":
                    score,

                "total":
                    total,

                "percentage":
                    percentage

            }


    # =====================================================
    # RENDER PAGE
    # =====================================================

    return render_template(

        "index.html",

        quiz=quiz,

        text=text,

        message=message,

        result=result

    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=False

    )
