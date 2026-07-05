"""
Generates a synthetic knowledge base of historical contact-center incident
summaries — the kind of unstructured text a RAG system retrieves from.
100% SYNTHETIC. Pairs with the structured data in banking-intent-classifier
and timeseries-forecasting-lab to cover text/RAG, tabular classification,
and time series in one coherent portfolio.
"""
import random
import pandas as pd

random.seed(42)

TEMPLATES = [
    ("volume_spike", "Call volume spiked {pct}% above forecast on {date} following {cause}. "
                      "Wait times increased to {wait} minutes. Root cause identified as {cause}. "
                      "Resolution: {resolution}."),
    ("system_outage", "Contact center experienced a {duration}-minute outage of the {system} system "
                       "on {date}. Approximately {tickets} tickets were delayed. "
                       "Resolution: {resolution}."),
    ("escalation_cluster", "A cluster of {count} escalations related to {topic} occurred on {date}, "
                            "concentrated among {segment} customers. "
                            "Resolution: {resolution}."),
    ("staffing_gap", "Staffing fell short by {agents} agents during the {shift} shift on {date}, "
                      "driven by {cause}. Average handle time increased by {pct}%. "
                      "Resolution: {resolution}."),
]

CAUSES = ["a billing statement release", "a product outage", "a marketing promotion",
          "a policy change announcement", "a payment processing delay", "seasonal demand"]
SYSTEMS = ["IVR", "CRM", "payment gateway", "authentication service", "callback queue"]
TOPICS = ["fee disputes", "fraud alerts", "app login issues", "loan application delays",
          "card replacement requests"]
SEGMENTS = ["new", "long-tenured", "mobile-app", "high-value"]
RESOLUTIONS = [
    "Additional agents were reassigned from overflow queue.",
    "Root cause was patched and monitoring alert threshold was lowered.",
    "Customers were proactively notified via SMS to reduce inbound volume.",
    "Escalation path was streamlined to reduce handle time.",
    "Staffing schedule was adjusted for the following week.",
]

def random_date():
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"2025-{month:02d}-{day:02d}"

def generate_incident_logs(n=150):
    rows = []
    for i in range(n):
        kind, template = random.choice(TEMPLATES)
        text = template.format(
            pct=random.randint(15, 85),
            date=random_date(),
            cause=random.choice(CAUSES),
            wait=random.randint(8, 35),
            resolution=random.choice(RESOLUTIONS),
            duration=random.randint(10, 90),
            system=random.choice(SYSTEMS),
            tickets=random.randint(50, 800),
            count=random.randint(10, 120),
            topic=random.choice(TOPICS),
            segment=random.choice(SEGMENTS),
            agents=random.randint(3, 20),
            shift=random.choice(["morning", "afternoon", "overnight", "weekend"]),
        )
        rows.append({"id": f"incident_{i:04d}", "type": kind, "text": text})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = generate_incident_logs()
    df.to_csv("incident_logs.csv", index=False)
    print(f"Generated {len(df)} synthetic incident log entries")
    print(df.head(3).to_string())
    print(f"\nSaved to incident_logs.csv")
