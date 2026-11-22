
from pathlib import Path

import joblib

# Load the trained Kannada-English model
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "vaaniguard_kannada_baseline.pkl")

print("\n===== VaaniGuard: Manual Message Testing =====")
print("Enter a payment-related message.")
print("Type 'exit' to stop.\n")

while True:
    message = input("Enter message: ").strip()

    if message.lower() == "exit":
        print("Testing completed!")
        break

    if not message:
        print("Please enter a valid message.\n")
        continue

    # Predict the message category
    prediction = model.predict([message])[0]

    # Get prediction probabilities
    probabilities = model.predict_proba([message])[0]

    classes = model.classes_

    print("\n----- VaaniGuard Result -----")
    print("Message:", message)
    print("Prediction:", prediction.upper())

    for label, probability in zip(classes, probabilities):
        print(f"{label.capitalize()} probability: {probability * 100:.2f}%")

    print("-----------------------------\n")
