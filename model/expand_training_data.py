
from pathlib import Path

import pandas as pd

# Load only the existing training data
BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "vaaniguard_scenario_heldout_split.csv")

train = df[df["split"] == "training"].copy()
evaluation = df[df["split"] == "evaluation"].copy()

# Never modify or add examples to evaluation data
new_examples = [
    # OTP - SCAM
    ("otp_direct_romanized", "UPI refund beku andre OTP nanage heli.", "romanized_kannada", "scam", "otp_request"),
    ("otp_direct_mixed", "Refund ಪಡೆಯಲು ನಿಮ್ಮ OTP share ಮಾಡಿ.", "kannada_english_mixed", "scam", "otp_request"),
    ("otp_direct_kannada", "ಹಣ ವಾಪಸ್ ಪಡೆಯಲು ನಿಮ್ಮ OTP ತಿಳಿಸಿ.", "kannada_script", "scam", "otp_request"),

    # OTP - LEGITIMATE
    ("otp_safe_romanized", "Bank yavattu OTP kelalla, yaarigu helbedi.", "romanized_kannada", "legitimate", "otp_safety"),
    ("otp_safe_mixed", "OTP ಯಾರಿಗೂ share ಮಾಡಬೇಡಿ ಎಂದು bank ಎಚ್ಚರಿಸಿದೆ.", "kannada_english_mixed", "legitimate", "otp_safety"),
    ("otp_safe_kannada", "ನಿಮ್ಮ ಬ್ಯಾಂಕ್ OTP ಅನ್ನು ಯಾರೊಂದಿಗೂ ಹಂಚಿಕೊಳ್ಳಬೇಡಿ.", "kannada_script", "legitimate", "otp_safety"),

    # REMOTE ACCESS - SCAM
    ("remote_scam_romanized", "Bank verification ge remote access app install maadi, screen share maadi.", "romanized_kannada", "scam", "remote_access_scam"),
    ("remote_scam_mixed", "Account verify ಮಾಡಲು remote access app install ಮಾಡಿ.", "kannada_english_mixed", "scam", "remote_access_scam"),
    ("remote_scam_kannada", "ಬ್ಯಾಂಕ್ ಪರಿಶೀಲನೆಗಾಗಿ ಈ ಅಪ್ಲಿಕೇಶನ್ ಸ್ಥಾಪಿಸಿ ಫೋನ್ ನಿಯಂತ್ರಣ ನೀಡಿ.", "kannada_script", "scam", "remote_access_scam"),

    # REMOTE ACCESS - LEGITIMATE
    ("remote_safe_romanized", "Unknown remote access app install madabedi anta bank helide.", "romanized_kannada", "legitimate", "remote_access_safety"),
    ("remote_safe_mixed", "Unknown remote access app install ಮಾಡಬೇಡಿ ಎಂದು bank ಸೂಚಿಸಿದೆ.", "kannada_english_mixed", "legitimate", "remote_access_safety"),
    ("remote_safe_kannada", "ಅಪರಿಚಿತರು ಹೇಳಿದ ರಿಮೋಟ್ ಆಕ್ಸೆಸ್ ಅಪ್ಲಿಕೇಶನ್ ಸ್ಥಾಪಿಸಬೇಡಿ.", "kannada_script", "legitimate", "remote_access_safety"),

    # QR - SCAM
    ("qr_scam_romanized", "Money receive madoke ee QR scan maadi UPI PIN haki.", "romanized_kannada", "scam", "qr_receive_scam"),
    ("qr_scam_mixed", "ಹಣ receive ಮಾಡಲು ಈ QR scan ಮಾಡಿ PIN enter ಮಾಡಿ.", "kannada_english_mixed", "scam", "qr_receive_scam"),
    ("qr_scam_kannada", "ಹಣ ಸ್ವೀಕರಿಸಲು ಈ QR ಕೋಡ್ ಸ್ಕ್ಯಾನ್ ಮಾಡಿ UPI ಪಿನ್ ನಮೂದಿಸಿ.", "kannada_script", "scam", "qr_receive_scam"),

    # QR - LEGITIMATE
    ("qr_safe_romanized", "Shop alli QR scan maadi bill pay madide.", "romanized_kannada", "legitimate", "normal_payment"),
    ("qr_safe_mixed", "Shop ನಲ್ಲಿ QR scan ಮಾಡಿ bill pay ಮಾಡಿದೆ.", "kannada_english_mixed", "legitimate", "normal_payment"),
    ("qr_safe_kannada", "ಅಂಗಡಿಯಲ್ಲಿ QR ಕೋಡ್ ಸ್ಕ್ಯಾನ್ ಮಾಡಿ ಬಿಲ್ ಪಾವತಿಸಿದೆ.", "kannada_script", "legitimate", "normal_payment"),

    # KYC - SCAM
    ("kyc_scam_romanized", "KYC expire agide, account block agutte, ee link click maadi.", "romanized_kannada", "scam", "kyc_urgency_link"),
    ("kyc_scam_mixed", "KYC pending ಇದೆ, account freeze ಆಗುತ್ತದೆ, link open ಮಾಡಿ.", "kannada_english_mixed", "scam", "kyc_urgency_link"),
    ("kyc_scam_kannada", "ನಿಮ್ಮ KYC ಅಪೂರ್ಣವಾಗಿದೆ, ಖಾತೆ ಉಳಿಸಲು ಈ ಲಿಂಕ್ ತೆರೆಯಿರಿ.", "kannada_script", "scam", "kyc_urgency_link"),

    # KYC - LEGITIMATE
    ("kyc_safe_romanized", "KYC update madoke bank branch ge hogtini.", "romanized_kannada", "legitimate", "official_kyc"),
    ("kyc_safe_mixed", "KYC update ಮಾಡಲು bank branch ಗೆ ಹೋಗುತ್ತೇನೆ.", "kannada_english_mixed", "legitimate", "official_kyc"),
    ("kyc_safe_kannada", "ನನ್ನ KYC ವಿವರಗಳನ್ನು ನವೀಕರಿಸಲು ಬ್ಯಾಂಕ್ ಶಾಖೆಗೆ ಹೋಗುತ್ತೇನೆ.", "kannada_script", "legitimate", "official_kyc"),
]

columns = [
    "id",
    "text",
    "language_format",
    "label",
    "scenario",
    "origin",
    "review_status",
    "split"
]

new_df = pd.DataFrame(new_examples, columns=[
    "id", "text", "language_format", "label", "scenario"
])

new_df["origin"] = "synthetic_expansion_new"
new_df["review_status"] = "needs_native_speaker_review"
new_df["split"] = "training"

# Ensure new IDs do not conflict
existing_ids = set(df["id"])
if any(item in existing_ids for item in new_df["id"]):
    raise ValueError("An example ID already exists.")

# Combine training data only
expanded_train = pd.concat(
    [train, new_df[columns]],
    ignore_index=True
)

# Keep evaluation data unchanged
expanded_dataset = pd.concat(
    [expanded_train, evaluation[columns]],
    ignore_index=True
)

# Save separate files
expanded_train.to_csv(
    BASE_DIR / "vaaniguard_expanded_training.csv",
    index=False
)

expanded_dataset.to_csv(
    BASE_DIR / "vaaniguard_expanded_scenario_dataset.csv",
    index=False
)

print("\nEXPANSION COMPLETED")
print("Original training examples:", len(train))
print("New synthetic examples:", len(new_df))
print("Expanded training examples:", len(expanded_train))
print("Evaluation examples:", len(evaluation))

print("\nTRAINING DISTRIBUTION")
print(pd.crosstab(
    expanded_train["language_format"],
    expanded_train["label"]
))

print("\nNEW EXAMPLES BY SCENARIO")
print(new_df.groupby(["scenario", "label"]).size())

print("\nEvaluation data preserved:", len(evaluation))
print("Saved: vaaniguard_expanded_training.csv")
print("Saved: vaaniguard_expanded_scenario_dataset.csv")
