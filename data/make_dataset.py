"""Generates the sample "private" dataset: an internal helpdesk for a fictional company.

The facts below are invented, so a base model cannot know them. After fine-tuning, the
model answers these questions correctly and signs off the way the company does, which
makes the before/after comparison in the notebook obvious.

Run after editing: python data/make_dataset.py
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
SIGN_OFF = "\n\n— Orbital Helpdesk"

# (answer, [question phrasings]); the last phrasing of each group is held out for eval.
FACTS = [
    ("We use Roastline to schedule roasts. Open it from the internal apps page and pick your roaster before booking a slot.",
     ["What tool do we use for roast scheduling?", "Where do I book a roasting slot?", "How do I schedule a roast?",
      "Which app handles the roast schedule?", "What's the name of our roast scheduling tool?"]),
    ("Inventory lives in Beanstalk. Green coffee, packaging and merch are all tracked there, and counts are reconciled every Monday morning.",
     ["Which system tracks our inventory?", "Where do I check how much green coffee we have?", "What is Beanstalk used for?",
      "How do I look up packaging stock?", "Where is inventory tracked?"]),
    ("The Cupping Room is our quality tool. Every production roast gets a cupping score logged there before it ships.",
     ["What is the Cupping Room?", "Where do cupping scores get logged?", "How is roast quality tracked?",
      "Which tool holds QA scores for roasts?", "Where do I record a cupping score?"]),
    ("Full-time employees get 24 days of paid time off per year, plus the ten company holidays.",
     ["How many PTO days do employees get?", "What's our vacation policy?", "How much paid time off do I have per year?",
      "How many days off do full-time staff get?", "What is the PTO allowance?"]),
    ("Submit expenses through Ledger Lane within 14 days of the purchase. Receipts over $25 must be attached.",
     ["How do I submit an expense report?", "Where do expenses go?", "What is the deadline for expense reports?",
      "Which tool do we use for expenses?", "How long do I have to file an expense?"]),
    ("The weekly roast schedule is posted every Thursday at 5pm in Roastline and mirrored to the #roasting channel.",
     ["When is the roast schedule posted?", "What day does the weekly roast plan come out?", "Where is the roast schedule announced?",
      "When can I see next week's roasting plan?", "What time is the roast schedule published?"]),
    ("Huila Reserve, our Colombian single origin, ships to subscribers every Tuesday.",
     ["When does the Huila Reserve ship?", "What day do Huila Reserve orders go out?", "Which day is the Colombian single origin shipped?",
      "When do subscribers get Huila Reserve?", "What's the shipping day for Huila Reserve?"]),
    ("Customer support is staffed from 7am to 6pm Pacific, Monday through Saturday. Outside those hours, tickets queue in Beanstalk Desk.",
     ["What are our support hours?", "When is customer support open?", "Until what time does support run?",
      "Is support open on Saturdays?", "What are the customer service hours?"]),
    ("Our CEO is Priya Raman. She founded Orbital Coffee Co. in Portland in 2019.",
     ["Who is our CEO?", "Who founded the company?", "Who runs Orbital Coffee?",
      "When was Orbital Coffee founded, and by whom?", "Who is the chief executive?"]),
    ("The roastery and office are at 118 Alder Street, Portland. Deliveries go to the loading dock on the north side.",
     ["What is the office address?", "Where is the roastery?", "Where should deliveries be sent?",
      "What's the street address of the company?", "Where is Orbital Coffee located?"]),
    ("The company mascot is Bean the Sloth. Bean appears on the subscription boxes and the loyalty stickers.",
     ["What is the company mascot?", "Who is Bean?", "What's on the subscription boxes?",
      "Do we have a mascot?", "What animal is our mascot?"]),
    ("The guest Wi-Fi network is OrbitalGuest and the password is on the whiteboard in the cupping room.",
     ["What is the guest wifi?", "How do visitors get on the wifi?", "What's the name of the guest network?",
      "Where is the wifi password?", "Which wifi should guests use?"]),
    ("Subscriptions renew on the 1st of each month. Members can pause up to three months a year from their account page.",
     ["When do subscriptions renew?", "Can members pause their subscription?", "What is the subscription renewal date?",
      "How long can a subscriber pause?", "When are subscribers billed?"]),
    ("Standard shipping is free on orders over $35. Below that it is a flat $6 within the United States.",
     ["What is the shipping policy?", "Is shipping free?", "How much does shipping cost?",
      "What's the free shipping threshold?", "Do we charge for shipping?"]),
    ("Returns are accepted within 30 days for unopened bags. Opened coffee is replaced, not refunded, if it arrived damaged.",
     ["What is the return policy?", "Can customers return coffee?", "How long do customers have to return an order?",
      "What happens if a bag arrives damaged?", "Do we refund opened bags?"]),
    ("The Q4 all-hands is on the second Friday of October at 10am in the cupping room, with remote staff on the usual call link.",
     ["When is the Q4 all-hands?", "What time is the October all-hands?", "Where is the all-hands held?",
      "When is the next company meeting?", "What day is the Q4 all hands?"]),
    ("Espresso Orbit is our flagship blend: 60% Brazil, 40% Ethiopia, roasted to a medium-dark profile.",
     ["What is in the Espresso Orbit blend?", "What's our flagship espresso?", "How is Espresso Orbit roasted?",
      "What beans go into Espresso Orbit?", "Describe the Espresso Orbit blend."]),
    ("New hires get their laptop and Roastline, Beanstalk and Ledger Lane accounts from IT on day one. Ask in #it-help if anything is missing.",
     ["What does a new hire get on day one?", "How do I get access to Roastline?", "Who sets up accounts for new employees?",
      "Where do I ask for missing tool access?", "How is onboarding handled?"]),
    ("Wholesale accounts order through the wholesale portal by Wednesday noon for delivery the following Monday.",
     ["What is the wholesale ordering deadline?", "How do wholesale customers order?", "When do wholesale orders get delivered?",
      "Where do cafes place wholesale orders?", "What's the cutoff for wholesale orders?"]),
    ("Green coffee arrives at the port of Portland and is trucked to the roastery on the first Monday of each month.",
     ["When does green coffee arrive?", "How does green coffee get to the roastery?", "Which port do we import through?",
      "How often do green coffee deliveries happen?", "When are green beans delivered?"]),
]


def example(question: str, answer: str) -> dict:
    return {"messages": [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer + SIGN_OFF},
    ]}


def main() -> None:
    random.seed(7)
    train, evaluation = [], []
    for answer, questions in FACTS:
        for question in questions[:-1]:
            train.append(example(question, answer))
        evaluation.append(example(questions[-1], answer))
    random.shuffle(train)
    (HERE / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in train))
    (HERE / "eval.jsonl").write_text("".join(json.dumps(row) + "\n" for row in evaluation))
    print(f"wrote {len(train)} training and {len(evaluation)} eval examples")


if __name__ == "__main__":
    main()
