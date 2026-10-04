from flask import Flask, render_template, request
import nltk
import re
import random

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLP data
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

app = Flask(__name__)

# Secret key for session handling
app.secret_key = "quiz-generator-secret-key"

stop_words = set(stopwords.words("english"))


# -----------------------------
# NLP KEYWORD EXTRACTION
# -----------------------------

def extract_keywords(text, num_keywords=30):

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=num_keywords
        )

        vectorizer.fit_transform([text])

        return list(vectorizer.get_feature_names_out())

    except Exception:
        return []


# -----------------------------
# QUESTION GENERATION
# -----------------------------

def generate_questions(text, num_questions=5):

    sentences = sent_tokenize(text)

    keywords = extract_keywords(text, 30)

    questions = []

    for sentence in sentences:

        sentence_clean = sentence.strip()

        if len(sentence_clean.split()) < 5:
            continue

        sentence_words = word_tokenize(
            sentence_clean.lower()
        )

        matching_keywords = [
            keyword
            for keyword in keywords
            if keyword.lower() in sentence_words
        ]

        if not matching_keywords:
            continue

        answer = matching_keywords[0]

        pattern = r"\b" + re.escape(answer) + r"\b"

        question_text = re.sub(
            pattern,
            "_____",
            sentence_clean,
            count=1,
            flags=re.IGNORECASE
        )

        if "_____" in question_text:

            questions.append({
                "question": "Fill in the blank: " + question_text,
                "answer": answer
            })


    # Remove duplicate questions

    unique_questions = []

    seen = set()

    for q in questions:

        if q["question"] not in seen:

            unique_questions.append(q)

            seen.add(q["question"])


    return unique_questions[:num_questions]


# -----------------------------
# MCQ GENERATION
# -----------------------------

def generate_mcqs(text, num_questions=5):

    questions = generate_questions(
        text,
        num_questions
    )

    all_keywords = extract_keywords(
        text,
        40
    )

    mcqs = []

    for q in questions:

        correct_answer = q["answer"]

        wrong_answers = [
            word
            for word in all_keywords
            if word.lower() != correct_answer.lower()
        ]

        wrong_answers = list(
            set(wrong_answers)
        )

        random.shuffle(wrong_answers)

        if len(wrong_answers) < 3:
            continue

        options = wrong_answers[:3] + [correct_answer]

        random.shuffle(options)

        mcqs.append({

            "question": q["question"],

            "options": options,

            "answer": correct_answer

        })

    return mcqs


# -----------------------------
# MAIN PAGE
# -----------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    quiz = []
    text = ""
    message = ""
    result = None

    # Generate quiz
    if request.method == "POST":

        action = request.form.get("action")

        # -------------------------
        # GENERATE QUIZ
        # -------------------------

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


        # -------------------------
        # SUBMIT QUIZ
        # -------------------------

        elif action == "submit":

            # Reconstruct quiz from hidden form data

            question_count = int(
                request.form.get(
                    "question_count",
                    0
                )
            )

            for i in range(question_count):

                question = request.form.get(
                    f"question_{i}"
                )

                correct_answer = request.form.get(
                    f"correct_{i}"
                )

                selected_answer = request.form.get(
                    f"answer_{i}"
                )

                quiz.append({

                    "question": question,

                    "answer": correct_answer,

                    "selected": selected_answer

                })


            score = 0

            for q in quiz:

                if (
                    q["selected"]
                    and
                    q["selected"].lower()
                    == q["answer"].lower()
                ):

                    score += 1


            result = {

                "score": score,

                "total": len(quiz),

                "percentage": round(
                    (score / len(quiz)) * 100
                ) if quiz else 0

            }


    return render_template(

        "index.html",

        quiz=quiz,

        text=text,

        message=message,

        result=result

    )


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
