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

app.secret_key = "quiz-generator-secret-key"

stop_words = set(stopwords.words("english"))


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):
    """Clean unnecessary whitespace."""

    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================================================
# TF-IDF KEYWORD EXTRACTION
# =========================================================

def extract_keywords(text, num_keywords=25):

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

        ranked = sorted(
            zip(terms, scores),
            key=lambda x: x[1],
            reverse=True
        )

        keywords = []

        for term, score in ranked:

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
    Extract likely concepts/topics from the beginning
    of informative sentences.

    Example:
    Artificial Intelligence is a branch...
    -> Artificial Intelligence

    Machine learning allows computers...
    -> Machine learning
    """

    sentences = sent_tokenize(text)

    concepts = []

    # Verbs commonly appearing after a subject/concept
    pattern = re.compile(
        r"^([A-Z][A-Za-z0-9-]*(?:\s+[A-Za-z0-9-]+){0,5})"
        r"\s+"
        r"(is|are|was|were|allows|enables|helps|provides|"
        r"supports|focuses|uses|can|refers|means|consists)\b",
        re.IGNORECASE
    )

    for sentence in sentences:

        sentence = sentence.strip()

        match = pattern.search(sentence)

        if match:

            concept = match.group(1).strip()

            # Remove trailing punctuation
            concept = re.sub(r"[.,:;]+$", "", concept)

            # Avoid extremely generic concepts
            if concept.lower() not in stop_words:

                if concept not in concepts:

                    concepts.append(concept)

    return concepts


# =========================================================
# BUILD CANDIDATE ANSWERS
# =========================================================

def get_candidate_answers(text):

    concepts = extract_concepts(text)

    keywords = extract_keywords(text, 30)

    candidates = []

    # Concepts get highest priority
    for concept in concepts:

        if concept.lower() not in [
            x.lower() for x in candidates
        ]:

            candidates.append(concept)

    # Add useful keywords as backup
    for keyword in keywords:

        # Avoid extremely short words
        if len(keyword) < 4:
            continue

        # Avoid obvious generic words
        if keyword.lower() in {
            "using",
            "used",
            "allows",
            "helps",
            "make",
            "makes",
            "focuses",
            "branch",
            "important",
            "part",
            "computer"
        }:
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


        # -------------------------------------------------
        # Pattern 1: X is / are ...
        # -------------------------------------------------

        match = re.match(
            r"^([A-Z][A-Za-z0-9-]*(?:\s+[A-Za-z0-9-]+){0,5})"
            r"\s+(is|are|was|were)\s+(.+)$",
            sentence
        )

        if match:

            subject = match.group(1).strip()
            predicate = match.group(3).strip()

            # Remove trailing punctuation
            predicate = predicate.rstrip(".!?")

            # Don't use overly long subjects
            if 1 <= len(subject.split()) <= 6:

                question = (
                    "Which concept is " +
                    predicate +
                    "?"
                )

                questions.append({
                    "question": question,
                    "answer": subject
                })

                continue


        # -------------------------------------------------
        # Pattern 2: X allows / enables / helps ...
        # -------------------------------------------------

        match = re.match(
            r"^([A-Z][A-Za-z0-9-]*(?:\s+[A-Za-z0-9-]+){0,5})"
            r"\s+(allows|enables|helps|provides|supports)\s+(.+)$",
            sentence,
            re.IGNORECASE
        )

        if match:

            subject = match.group(1).strip()
            verb = match.group(2).strip()
            rest = match.group(3).strip()

            rest = rest.rstrip(".!?")

            if 1 <= len(subject.split()) <= 6:

                question = (
                    "What " +
                    verb +
                    " " +
                    rest +
                    "?"
                )

                questions.append({
                    "question": question,
                    "answer": subject
                })

                continue


        # -------------------------------------------------
        # Pattern 3: fallback keyword question
        # -------------------------------------------------

        keywords = extract_keywords(sentence, 10)

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
                    "answer": answer
                })


    # =====================================================
    # REMOVE DUPLICATES
    # =====================================================

    unique_questions = []

    seen = set()

    for q in questions:

        key = q["question"].lower()

        if key not in seen:

            unique_questions.append(q)
            seen.add(key)


    # Shuffle so the same questions aren't always first
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


    for q in questions:

        correct_answer = q["answer"]


        # -------------------------------------------------
        # Find suitable distractors
        # -------------------------------------------------

        possible_distractors = []

        for candidate in candidates:

            if (
                candidate.lower()
                != correct_answer.lower()
            ):

                # Avoid answers that are too similar
                if (
                    candidate.lower()
                    not in correct_answer.lower()
                    and
                    correct_answer.lower()
                    not in candidate.lower()
                ):

                    possible_distractors.append(candidate)


        # Shuffle distractors
        random.shuffle(possible_distractors)


        # Add fallback keywords if needed
        if len(possible_distractors) < 3:

            keywords = extract_keywords(
                text,
                40
            )

            for keyword in keywords:

                if (
                    keyword.lower()
                    != correct_answer.lower()
                    and
                    keyword.lower()
                    not in [
                        x.lower()
                        for x in possible_distractors
                    ]
                ):

                    possible_distractors.append(
                        keyword
                    )

                if len(possible_distractors) >= 3:
                    break


        if len(possible_distractors) < 3:
            continue


        distractors = possible_distractors[:3]


        # -------------------------------------------------
        # Create options
        # -------------------------------------------------

        options = distractors + [
            correct_answer
        ]

        random.shuffle(options)


        mcqs.append({

            "question": q["question"],

            "options": options,

            "answer": correct_answer

        })


    return mcqs


# =========================================================
# HOME PAGE
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

            except:

                num_questions = 5


            if text.strip():

                quiz = generate_mcqs(
                    text,
                    num_questions
                )


                if not quiz:

                    message = (
                        "Not enough suitable "
                        "content to generate questions."
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

            except:

                question_count = 0


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

                    "question": question,

                    "answer": correct_answer,

                    "selected": selected_answer

                })


            # Calculate score

            score = 0

            for q in quiz:

                if (

                    q["selected"]

                    and

                    q["selected"].lower()
                    == q["answer"].lower()

                ):

                    score += 1


            total = len(quiz)


            percentage = (
                round(
                    (score / total) * 100
                )
                if total > 0
                else 0
            )


            result = {

                "score": score,

                "total": total,

                "percentage": percentage

            }


    return render_template(

        "index.html",

        quiz=quiz,

        text=text,

        message=message,

        result=result

    )


# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
