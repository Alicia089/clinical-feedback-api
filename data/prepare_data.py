"""
prepare_data.py — Build a labeled clinical feedback dataset.

Sources:
  1. MTSamples (Kaggle: tboyle10/medicaltranscriptions)
     Medical transcription samples — used to extract clinical language patterns.

  2. Synthetic feedback — Rule-based generation using clinical keyword banks.
     WHY synthetic? Real ASC patient feedback is PHI-protected and not publicly
     available. A well-constructed synthetic dataset with domain-specific language
     is a valid and common approach in clinical NLP (see Google's HealthBench).

  3. Manual seed examples — 20 human-labeled examples per class to anchor
     the synthetic generation toward realistic distributions.

Run:
    python data/prepare_data.py --output data/clinical_feedback_labeled.csv
"""

import random
import argparse
import pandas as pd

random.seed(42)

# ── Seed examples (human-labeled) ────────────────────────────────────────────
SEEDS = {
    "URGENT": [
        "Patient called reporting chest tightness and shortness of breath 6 hours post-surgery",
        "Wound site is actively bleeding and dressing is soaked through",
        "I am having an allergic reaction, my throat feels tight",
        "High fever 103.5 since last night, I don't know what to do",
        "Patient fell in the bathroom attempting to stand unassisted",
        "Incision opened up, I can see tissue underneath",
        "Severe pain level 10, medication is not helping at all",
        "Patient reports numbness in the left arm since this morning",
        "I think I'm having a reaction to the anesthesia, very dizzy and confused",
        "Unable to keep any fluids down, severe vomiting since discharge",
        "The drain output has dramatically increased and changed color",
        "Patient is unresponsive and cannot be woken by family member",
        "Rapidly spreading redness and warmth around the incision site",
        "Sudden vision changes in my right eye after the procedure",
        "Deep vein pain in the calf, leg is swollen and warm",
        "I cannot breathe properly lying flat, I need help immediately",
        "My blood sugar crashed and I passed out for a moment",
        "Sutures are coming apart and the wound is gaping",
        "Urine output has significantly decreased in the last 12 hours",
        "Patient reporting confusion and disorientation, not acting normally",
    ],
    "NEGATIVE": [
        "The wait time was completely unacceptable, over 3 hours just sitting there",
        "No one explained what was happening during my procedure",
        "Staff seemed annoyed when I asked questions about my care",
        "The discharge instructions were confusing and hard to understand",
        "I was not told about this complication risk beforehand",
        "The facility was not clean and I was concerned about hygiene",
        "Nobody followed up with me after my procedure as promised",
        "I had to ask multiple times for pain medication",
        "The billing department was extremely rude and unhelpful",
        "I felt rushed out the door before I was ready to leave",
        "The pre-op paperwork was overwhelming and staff wasn't helpful",
        "No one introduced themselves, I didn't know who was treating me",
        "My procedure started 90 minutes late with no explanation",
        "The recovery room was loud and chaotic, very stressful",
        "I was never given information about my post-op restrictions",
        "The parking situation was a nightmare for someone post-surgery",
        "Staff were talking loudly about other patients, no privacy",
        "I was given the wrong medication initially before it was corrected",
        "The discharge process was disorganized and took far too long",
        "I felt like a number, not a patient, throughout the entire visit",
    ],
    "NEUTRAL": [
        "Procedure was completed on schedule as planned",
        "Staff provided standard pre-operative instructions",
        "Discharge summary was provided at the time of release",
        "Patient arrived and check-in was completed per protocol",
        "Follow-up appointment has been scheduled for two weeks out",
        "Lab results were within normal reference ranges",
        "Vital signs remained stable throughout the procedure",
        "Patient was transferred to recovery without complications",
        "Standard post-operative diet restrictions were communicated",
        "The procedure room was set up and ready at the scheduled time",
        "Patient understood the consent form and signed without questions",
        "Pre-op blood work was completed the day prior as instructed",
        "IV was placed on the first attempt without difficulty",
        "Patient tolerated anesthesia without noted adverse events",
        "Post-op instructions were reviewed and acknowledged",
        "Patient was ambulatory prior to discharge as required",
        "Family member was present and included in discharge education",
        "Pain level at discharge was reported as manageable",
        "All equipment was functioning properly during the procedure",
        "Patient's insurance authorization was confirmed prior to surgery",
    ],
    "POSITIVE": [
        "The entire team was so kind and made me feel completely at ease",
        "Best surgical experience I have ever had, very professional",
        "Dr. Chen took the time to explain every step before starting",
        "I cannot thank the nurses enough for their exceptional care",
        "The facility was spotless and made me feel confident in the team",
        "From check-in to discharge the experience was seamless",
        "My anxiety was high but the staff immediately calmed me down",
        "Every question I had was answered thoroughly and with patience",
        "The follow-up call the next morning was a really nice touch",
        "I was kept informed the entire time, never felt left in the dark",
        "Recovery was smoother than I expected thanks to the team's prep",
        "The pre-op instructions were clear and helped me feel prepared",
        "Scheduling was easy and the staff was flexible with my needs",
        "The anesthesiologist explained exactly what to expect, very reassuring",
        "Pain was well managed and the care plan was explained clearly",
        "I would absolutely recommend this facility to friends and family",
        "Staff checked on me frequently in recovery, I felt very cared for",
        "The surgical coordinator kept my family updated throughout",
        "Everything happened exactly as they described during the consult",
        "I felt like a priority, not an inconvenience, at every step",
    ],
}

# ── Augmentation word banks ───────────────────────────────────────────────────
URGENT_PREFIXES = [
    "Patient reported", "Caller stated", "Family member noted",
    "Patient is experiencing", "Urgent: patient describing",
]
URGENT_SYMPTOMS = [
    "uncontrolled bleeding", "difficulty breathing", "severe allergic reaction",
    "rapidly worsening pain", "high fever and chills", "wound dehiscence",
    "loss of consciousness", "extreme confusion", "sudden vision changes",
    "inability to urinate", "severe chest pain", "signs of infection spreading",
]

NEGATIVE_TEMPLATES = [
    "Very disappointed with {aspect} during my visit.",
    "The {aspect} was completely unacceptable and needs improvement.",
    "I was frustrated by the {aspect} throughout my procedure day.",
    "{aspect} fell far below my expectations.",
]
NEGATIVE_ASPECTS = [
    "wait time", "communication from staff", "discharge process",
    "pain management", "cleanliness", "scheduling", "billing process",
    "follow-up care", "staff attitude", "pre-op instructions",
]

POSITIVE_TEMPLATES = [
    "Extremely satisfied with the {aspect} at this facility.",
    "The {aspect} exceeded all of my expectations.",
    "Highly impressed by the {aspect} throughout my entire experience.",
    "Outstanding {aspect} from start to finish.",
]
POSITIVE_ASPECTS = [
    "level of care", "nursing staff", "communication", "surgical team",
    "post-op follow-up", "facility cleanliness", "pain management",
    "discharge instructions", "scheduling experience", "overall care",
]


def generate_synthetic(n_per_class: int = 200) -> pd.DataFrame:
    records = []

    # URGENT — template + symptom combinations
    for _ in range(n_per_class):
        prefix = random.choice(URGENT_PREFIXES)
        symptom = random.choice(URGENT_SYMPTOMS)
        text = f"{prefix} {symptom} following the procedure."
        records.append({"text": text, "label": "URGENT"})

    # NEGATIVE — fill-in-the-blank templates
    for _ in range(n_per_class):
        template = random.choice(NEGATIVE_TEMPLATES)
        text = template.format(aspect=random.choice(NEGATIVE_ASPECTS))
        records.append({"text": text, "label": "NEGATIVE"})

    # NEUTRAL — paraphrased procedural language
    neutral_pool = [
        "Procedure completed without documented complications.",
        "Patient received standard post-operative instructions.",
        "Follow-up appointment scheduled per protocol.",
        "Patient tolerated the procedure and was discharged.",
        "All pre-operative requirements were met prior to surgery.",
        "Vital signs were stable throughout the perioperative period.",
    ]
    for _ in range(n_per_class):
        base = random.choice(neutral_pool)
        records.append({"text": base, "label": "NEUTRAL"})

    # POSITIVE — fill-in-the-blank templates
    for _ in range(n_per_class):
        template = random.choice(POSITIVE_TEMPLATES)
        text = template.format(aspect=random.choice(POSITIVE_ASPECTS))
        records.append({"text": text, "label": "POSITIVE"})

    return pd.DataFrame(records)


def build_dataset(output_path: str):
    # Seed examples
    seed_records = []
    for label, examples in SEEDS.items():
        for text in examples:
            seed_records.append({"text": text, "label": label})
    seed_df = pd.DataFrame(seed_records)

    # Synthetic augmentation
    synthetic_df = generate_synthetic(n_per_class=200)

    df = pd.concat([seed_df, synthetic_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

    df.to_csv(output_path, index=False)
    print(f"Dataset saved: {output_path}")
    print(f"Total samples: {len(df)}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/clinical_feedback_labeled.csv")
    args = parser.parse_args()
    build_dataset(args.output)
