from flask import Flask, render_template, request
from agents.graph import build_graph

app = Flask(__name__)
graph = build_graph()


@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    sources = []
    question = ""

    if request.method == "POST":
        question = request.form.get("question", "").strip()

        if question:
            result = graph.invoke({
                "question": question,
                "retrieved_chunks": [],
                "draft_answer": "",
                "validation_result": False,
                "retry_count": 0
            })

            answer = result["draft_answer"]
            sources = result["retrieved_chunks"]

    return render_template(
        "index.html",
        question=question,
        answer=answer,
        sources=sources
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
