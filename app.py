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
    """Clean extra spaces and new lines."""

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================================================
# TF-IDF KEYWORD EXTRACTION
# =========================================================

def extract_keywords(text, num_keywords=30):
    """
    Extract important words and short phrases using TF-IDF.
    """

    text = clean_text(text)

    if not text:
        return []

    sentences = sent_tokenize(text)

    if not sentences:
        return []

    try:

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
    Try to detect meaningful concepts from sentences.

    Examples:
        Artificial Intelligence is...
        Machine learning allows...
        Natural language processing enables...
    """

    sentences = sent_tokenize(clean_text(text))

    concepts = []

    generic_words = {
        "this",
        "that",
        "it",
        "they",
        "he",
        "she",
        "these",
        "those"
    }

    patterns = [

        # X is/are...
        re.compile(
            r"^([A-Z][A-Za-z0-9-]*(?:\s+[A-Za-z0-9-]+){0,5})"
            r"\s+(?:is|are|was|were)\b"
        ),

        # X allows/enables/helps...
        re.compile(
            r"^([A-Z][A-Za-z0-9-]*(?:\s+[A-Za-z0-9-]+){0,5})"
            r"\s+(?:allows|enables|helps|provides|supports)\b",
            re.IGNORECASE
        )
    ]

    for sentence in sentences:

        sentence = sentence.strip()

        for pattern in patterns:

            match = pattern.search(sentence)

            if match:

                concept = match.group(1).strip()

                concept = re.sub(
                    r"[.,:;!?]+$",
                    "",
                    concept
                )

                if concept.lower() in generic_words:
                    continue

                if not 1 <= len(concept.split()) <= 6:
                    continue

                # Avoid duplicates
                if concept.lower() not in [
                    c.lower() for c in concepts
                ]:
                    concepts.append(concept)

                break

    return concepts


# =========================================================
# BUILD CANDIDATE ANSWERS
# =========================================================

def get_candidate_answers(text):

    concepts = extract_concepts(text)

    keywords = extract_keywords(text, 40)

    candidates = []

    # Prefer meaningful concepts
    for concept in concepts:

        if concept.lower() not in [
            x.lower() for x in candidates
        ]:

            candidates.append(concept)

    # Add useful keywords
    ignored_words = {
        "using",
        "used",
        "allows",
        "enable",
        "enables",
        "helps",
        "provides",
        "supports",
        "make",
        "makes",
        "making",
        "focuses",
        "branch",
        "important",
        "part",
        "computer",
        "computers",
        "technology",
        "field"
    }

    for keyword in keywords:

        if len(keyword) < 4:
            continue

        if keyword.lower() in ignored_words:
            continue

        if keyword.lower() not in [
            x.lower() for x in candidates
        ]:

            candidates.append(keyword)

    return candidates


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
        # PATTERN 1: "X is widely used in Y"
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

            fields = match.group(2).strip()

            fields = fields.rstrip(".!?")

            if 1 <= len(subject.split()) <= 6:

                questions.append({

                    "question":
                        f"Where is {subject} widely used?",

                    "answer":
                        subject

                })

                continue


        # =================================================
        # PATTERN 2: "X is a/an Y"
        # =================================================

        match = re.match(
            r"^([A-Z][A-Za-z0-9-]*(?:\s+[A-Za-z0-9-]+){0,5})"
            r"\s+(?:is|are)\s+"
            r"(?:a|an)\s+(.+)$",
            sentence
        )

        if match:

            subject = match.group(1).strip()

            description = match.group(2).strip()

            description = description.rstrip(".!?")

            if 1 <= len(subject.split()) <= 6:

                questions.append({

                    "question":
                        f"What is {subject}?",

                    "answer":
                        subject

                })

                continue


        # =================================================
        # PATTERN 3: "X is..."
        # =================================================

        match = re.match(
            r"^([A-Z][A-Za-z0-9-]*(?:\s+[A-Za-z0-9-]+){0,5})"
            r"\s+(?:is|are|was|were)\s+(.+)$",
            sentence
        )

        if match:

            subject = match.group(1).strip()

            description = match.group(2).strip()

            description = description.rstrip(".!?")

            if 1 <= len(subject.split()) <= 6:

                questions.append({

                    "question":
                        f"What is {subject}?",

                    "answer":
                        subject

                })

                continue


        # =================================================
        # PATTERN 4: "X allows..."
        # =================================================

        match = re.match(
            r"^([A-Z][A-Za-z0-9-]*(?:\s+[A-Za-z0-9-]+){0,5})"
            r"\s+(allows|enables|helps)\s+(.+)$",
            sentence,
            re.IGNORECASE
        )

        if match:

            subject = match.group(1).strip()

            verb = match.group(2).strip()

            action = match.group(3).strip()

            action = action.rstrip(".!?")

            if 1 <= len(subject.split()) <= 6:

                if verb.lower() == "allows":

                    question = (
                        f"What does {subject} allow?"
                    )

                elif verb.lower() == "enables":

                    question = (
                        f"What does {subject} enable?"
                    )

                else:

                    question = (
                        f"What does {subject} help with?"
                    )

                questions.append({

                    "question": question,

                    "answer": subject

                })

                continue


        # =================================================
        # PATTERN 5: "X focuses on..."
        # =================================================

        match = re.match(
            r"^([A-Z][A-Za-z0-9-]*(?:\s+[A-Za-z0-9-]+){0,5})"
            r"\s+focuses on\s+(.+)$",
            sentence,
            re.IGNORECASE
        )

        if match:

            subject = match.group(1).strip()

            purpose = match.group(2).strip()

            purpose = purpose.rstrip(".!?")

            if 1 <= len(subject.split()) <= 6:

                questions.append({

                    "question":
                        f"What is the main focus of {subject}?",

                    "answer":
                        subject

                })

                continue


        # =================================================
        # PATTERN 6: "X refers to..."
        # =================================================

        match = re.match(
            r"^([A-Z][A-Za-z0-9-]*(?:\s+[A-Za-z0-9-]+){0,5})"
            r"\s+refers to\s+(.+)$",
            sentence,
            re.IGNORECASE
        )

        if match:

            subject = match.group(1).strip()

            meaning = match.group(2).strip()

            meaning = meaning.rstrip(".!?")

            if 1 <= len(subject.split()) <= 6:

                questions.append({

                    "question":
                        f"What does {subject} refer to?",

                    "answer":
                        subject

                })

                continue


        # =================================================
        # FALLBACK: KEYWORD-BASED QUESTION
        # =================================================

        keywords = extract_keywords(
            sentence,
            num_keywords=10
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
                        "Complete the statement: " +
                        replaced,

                    "answer":
                        answer

                })


    # =====================================================
    # REMOVE DUPLICATE QUESTIONS
    # =====================================================

    unique_questions = []

    seen = set()

    for question in questions:

        key = question["question"].lower()

        if key not in seen:

            unique_questions.append(question)

            seen.add(key)


    # Shuffle question order
    random.shuffle(unique_questions)

    return unique_questions[:num_questions]


# =========================================================
# MCQ GENERATION
# =========================================================

def generate_mcqs(text, num_questions=5):

    questions = generate_questions(
        text,
        num_questions
    )

    candidates = get_candidate_answers(text)

    mcqs = []


    for question_data in questions:

        correct_answer = question_data["answer"]

        possible_distractors = []


        # -------------------------------------------------
        # Use other concepts as distractors
        # -------------------------------------------------

        for candidate in candidates:

            if candidate.lower() == correct_answer.lower():
                continue

            # Avoid extremely similar answers
            if (
                candidate.lower() in correct_answer.lower()
                or
                correct_answer.lower() in candidate.lower()
            ):
                continue

            if candidate not in possible_distractors:

                possible_distractors.append(candidate)


        random.shuffle(possible_distractors)


        # -------------------------------------------------
        # Backup keywords
        # -------------------------------------------------

        if len(possible_distractors) < 3:

            keywords = extract_keywords(
                text,
                50
            )

            for keyword in keywords:

                if keyword.lower() == correct_answer.lower():
                    continue

                if keyword.lower() in [
                    x.lower()
                    for x in possible_distractors
                ]:
                    continue

                if (
                    keyword.lower()
                    in correct_answer.lower()
                    or
                    correct_answer.lower()
                    in keyword.lower()
                ):
                    continue

                possible_distractors.append(
                    keyword
                )

                if len(possible_distractors) >= 3:
                    break


        # -------------------------------------------------
        # Skip if we cannot create 4 options
        # -------------------------------------------------

        if len(possible_distractors) < 3:
            continue


        distractors = possible_distractors[:3]

        options = distractors + [
            correct_answer
        ]


        # Shuffle answer position
        random.shuffle(options)


        mcqs.append({

            "question":
                question_data["question"],

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
    # POST REQUEST
    # =====================================================

    if request.method == "POST":

        action = request.form.get("action")


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


            # Limit questions
            num_questions = max(
                1,
                min(num_questions, 10)
            )


            if text.strip():

                quiz = generate_mcqs(
                    text,
                    num_questions
                )


                if not quiz:

                    message = (
                        "Not enough suitable content "
                        "to generate questions. "
                        "Try using a longer article."
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


            # -------------------------------------------------
            # Reconstruct submitted questions
            # -------------------------------------------------

            for i in range(question_count):

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


            # -------------------------------------------------
            # Calculate score
            # -------------------------------------------------

            score = 0


            for question in quiz:

                selected = question["selected"]

                correct = question["answer"]


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
