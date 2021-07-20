"""
This is a Python example on how to create a custom data enrichment component using Python and Flask.

For more information, see the Metamaze documentation or contact us at support@metamaze.eu.
"""

import logging
import pickle
from functools import wraps
from typing import Any, Callable

import numpy as np
from flask import Flask, abort, jsonify, request
from flask_cors import CORS
from laserembeddings import Laser
from sklearn.ensemble import RandomForestClassifier

from config import BEARER_TOKEN

app = Flask(__name__)

CORS(app, support_credentials=True)

PREDICTION_THRESHOLD = 0.35  # TODO fine-tune PREDICTION_THRESHOLD based on a final training


def load_classifier() -> RandomForestClassifier:
    with open("classifier.pkl", "rb") as f:
        classifier = pickle.load(f)
    return classifier


def create_app() -> Flask:

    app.classifier = load_classifier()
    app.embeddings_model = Laser()

    # classifier predicts based on the order of below classes
    app.ordered_classes = [
        "AG51",
        "AG14",
        "SC37",
        "RL27",
        "KM3",
        "AG48",
        "AG52",
        "RL4",
        "SP11",
        "KM61",
        "KM37",
        "HL38",
        "RL3",
        "AL4",
        "KM53",
    ]

    return app


def check_token(f: Callable) -> Callable:
    @wraps(f)
    def decorated(*args: Any, **kwargs: Any) -> Any:
        print("headers: ", request.headers)
        try:
            token_header = request.headers["authorization"]
        except KeyError:
            return abort(403, description="Authentication header missing.")

        auth_token = token_header.split(maxsplit=1)[1]
        if auth_token != BEARER_TOKEN:
            logging.error(f"Bad token '{auth_token}'")
            return abort(403, description="Incorrect or no authentication token.")

        return f(*args, **kwargs)

    return decorated


@app.route("/api/pred-entity-class", methods=["GET"])
@check_token
def predict_class_from_entity() -> Any:
    content = request.json

    enrichments = []

    for entity in content["entities"]:
        embeddings = app.embeddings_model.embed_sentences(entity["text"], lang=entity["language"])
        pred_probs = app.classifier.predict_proba(embeddings)[0]

        pred_index = np.argmax(pred_probs)
        if pred_probs[pred_index] > PREDICTION_THRESHOLD:
            pred_class = app.ordered_classes[pred_index]
        else:
            pred_class = "Unknown Entity"

        enrichments.append({"name": "class", "value": pred_class})

    return jsonify({"enrichments": enrichments})


@app.route("/", methods=["GET"])
def get_health() -> Any:
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    # do NOT run this in prod, flask dev server is not meant for that
    # check the dockerfile which starts a gunicorn server
    create_app().run(host="0.0.0.0", debug=True)
