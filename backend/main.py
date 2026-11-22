
import re
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI(
    title="VaaniGuard API",
    description="Kannada-English digital payment scam detection API",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "model" / "vaaniguard_kannada_baseline.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("VaaniGuard ML model loaded successfully!")
except Exception as error:
    raise RuntimeError(f"Could not load ML model: {error}")


class MessageRequest(BaseModel):
    message: str = Field(min_length=5, max_length=1000)


# Rule-based supporting indicators.
# These are not direct explanations of the ML model.

INDICATOR_GROUPS = {
    "OTP or PIN request": [
        "otp",
        "one time password",
        "upi pin",
        "pin",
        "password",
        "ಪಿನ್",
        "ಒಟಿಪಿ",
    ],
    "KYC or account verification": [
        "kyc",
        "verify your account",
        "account verification",
        "update kyc",
        "ವೈಕೆಸಿ",
    ],
    "Urgency or account threat": [
        "urgent",
        "immediately",
        "right now",
        "account blocked",
        "account suspended",
        "will be blocked",
        "ತಕ್ಷಣ",
        "ಬ್ಲಾಕ್",
    ],
    "Suspicious links or payment requests": [
        "click here",
        "payment link",
        "scan qr",
        "scan the qr",
        "refund",
        "collect request",
        "ನಿಮ್ಮ ಲಿಂಕ್",
    ],
    "Prize or reward claims": [
        "you won",
        "winner",
        "prize",
        "lottery",
        "reward",
        "cashback",
    ],
}

SAFETY_RECOMMENDATIONS = [
    "Never share your OTP, UPI PIN or banking password.",
    "Do not click unknown payment or verification links.",
    "Verify account-related messages through your bank's official app or website.",
    "Never approve an unknown UPI collect request.",
]


def phrase_matches(message, phrase):
    """
    Match English phrases as complete words.
    Match Kannada phrases without English word boundaries.
    """
    if re.search(r"[a-zA-Z0-9]", phrase):
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(phrase) + r"(?![a-zA-Z0-9])"
        return re.search(pattern, message, flags=re.IGNORECASE) is not None

    return phrase in message


def explain_message(message, prediction):
    detected_indicators = []

    for category, phrases in INDICATOR_GROUPS.items():
        matched_phrases = [
            phrase
            for phrase in phrases
            if phrase_matches(message, phrase)
        ]

        if matched_phrases:
            detected_indicators.append({
                "category": category,
                "matched_phrases": matched_phrases,
            })

    if detected_indicators:
        explanation = (
            "The message contains potentially suspicious language "
            "or payment-related patterns. Review the highlighted "
            "indicators carefully."
        )
    else:
        explanation = (
            "No configured suspicious phrases were identified. "
            "This does not guarantee that the message is legitimate."
        )

    if str(prediction).lower() == "scam":
        explanation += (
            " The experimental ML model classified this message as a scam."
        )
    else:
        explanation += (
            " The experimental ML model classified this message as legitimate. "
            "Remain cautious even when a message receives this prediction."
        )

    return detected_indicators, explanation


@app.get("/")
def home():
    return {
        "message": "Welcome to VaaniGuard API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/analyze")
def analyze_message(request: MessageRequest):
    cleaned_message = request.message.strip()

    if not cleaned_message:
        raise HTTPException(
            status_code=400,
            detail="Please enter a message to analyze.",
        )

    try:
        prediction = model.predict([cleaned_message])[0]

        probabilities = model.predict_proba([cleaned_message])[0]
        classes = list(model.classes_)

        probability_map = {
            str(label): float(probability) * 100
            for label, probability in zip(classes, probabilities)
        }

        scam_probability = probability_map.get("scam", 0.0)
        legitimate_probability = probability_map.get("legitimate", 0.0)

        indicators, explanation = explain_message(
            cleaned_message,
            prediction,
        )

        return {
            "message": cleaned_message,
            "prediction": str(prediction),
            "confidence": round(float(max(probabilities)) * 100, 2),
            "probabilities": {
                "legitimate": round(legitimate_probability, 2),
                "scam": round(scam_probability, 2),
            },
            "indicators": indicators,
            "explanation": explanation,
            "recommendations": SAFETY_RECOMMENDATIONS,
            "model": "vaaniguard_kannada_baseline",
            "status": "success",
            "notice": (
                "Experimental baseline ML prediction. "
                "The displayed indicators are rule-based supporting "
                "signals, not a direct explanation of the model's "
                "internal decision. This result is not a guarantee "
                "of message safety."
            ),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Message analysis failed: {str(error)}",
        )
