import json
from typing import Dict
from llama_cpp.llama_types import (
    SystemOneQuestion,
    SystemOneResponse,
)


def test_system_one_types():
    questions: Dict[str, SystemOneQuestion] = {
        "intent": {
            "type": "choice",
            "instructions": "Identify the primary topic.",
            "criteria": {
                "shipping": "Lost, delayed, or missing packages",
                "refund": "Requesting money back",
            },
        },
        "is_urgent": {
            "type": "noul",
            "instructions": "Is this urgent?",
        },
        "severity": {
            "type": "score",
            "instructions": "Rate severity 0 to 2.",
            "criteria": ["Low", "Medium", "High"],
        },
    }

    payload = json.dumps({"state": "My order is delayed", "questions": questions})
    loaded = json.loads(payload)
    assert loaded["questions"]["intent"]["type"] == "choice"
    assert loaded["questions"]["is_urgent"]["type"] == "noul"
    assert loaded["questions"]["severity"]["type"] == "score"


def test_system_one_response_structure():
    response: SystemOneResponse = {
        "model": "d1-Omni-2B-GGUF",
        "answers": {
            "intent": {
                "type": "choice",
                "choice": "shipping",
                "confidence": 0.94,
                "probabilities": {"shipping": 0.94, "refund": 0.06},
            },
            "is_urgent": {
                "type": "noul",
                "noul": 0.82,
            },
            "severity": {
                "type": "score",
                "score": 1.2,
                "confidence": 0.7,
                "legend": {"0": "Low", "1": "Medium", "2": "High"},
                "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3},
            },
        },
        "usage": {
            "input_tokens": 85,
            "output_tokens": 0,
        },
    }
    assert response["answers"]["intent"]["choice"] == "shipping"
    assert response["answers"]["is_urgent"]["noul"] == 0.82
    assert response["usage"]["output_tokens"] == 0


if __name__ == "__main__":
    test_system_one_types()
    test_system_one_response_structure()
    print("All System One unit tests passed successfully!")
