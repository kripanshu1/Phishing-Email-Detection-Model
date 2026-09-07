"""
dataset_generator.py
=====================
Generates a synthetic but realistic labeled dataset of phishing and
legitimate emails, so the project can be trained and demoed without
needing to download an external dataset.

If you have a real dataset (e.g. from Kaggle's "Phishing Email
Dataset" or the Nazario phishing corpus), you can skip this and load
your own CSV instead -- it just needs a `text` column (subject + body)
and a `label` column (`1` = phishing, `0` = legitimate). See
`load_dataset()` in phishing_detector.py.
"""

import random
import pandas as pd

random.seed(42)

# --------------------------------------------------------------------------
# Building blocks for PHISHING emails
# --------------------------------------------------------------------------

PHISHING_SUBJECTS = [
    "Urgent: Your Account Will Be Suspended",
    "Action Required: Verify Your Account Now",
    "You've Won a $1000 Gift Card!",
    "Security Alert: Unusual Sign-in Activity Detected",
    "Your Payment Failed - Update Billing Info Immediately",
    "Final Notice: Your Package Could Not Be Delivered",
    "Congratulations! You Are Our Lucky Winner",
    "Your PayPal Account Has Been Limited",
    "IRS Tax Refund Pending - Claim Now",
    "Your Password Will Expire Today - Reset Now",
    "Confirm Your Identity to Avoid Account Closure",
    "Unusual Login Attempt on Your Bank Account",
    "Your Netflix Subscription Payment Was Declined",
    "Claim Your Free iPhone Before It's Too Late",
    "Immediate Action Needed: Verify Your Email",
]

PHISHING_BODIES = [
    "Dear Customer, we have detected suspicious activity on your account. "
    "Click here {url} immediately to verify your identity or your account "
    "will be permanently suspended within 24 hours.",

    "Congratulations!!! You have been selected to receive a reward of $1000. "
    "To claim your prize, please click the link below and enter your bank "
    "details: {url}",

    "Your account has been temporarily limited due to unusual activity. "
    "Please confirm your information by clicking {url} within 48 hours to "
    "restore full access.",

    "We were unable to process your recent payment. Please update your "
    "billing details urgently at {url} to avoid service interruption.",

    "Act now! Your password expires today. Verify your credentials "
    "immediately at {url} or you will lose access to your account.",

    "Dear user, our records show a failed delivery attempt for your "
    "package. Click {url} and pay a small redelivery fee to reschedule.",

    "URGENT: Unusual sign-in activity was detected from a new device. "
    "If this wasn't you, secure your account now at {url}.",

    "You have 1 unclaimed reward waiting. Verify your identity at {url} "
    "to receive your gift card before it expires tonight.",

    "This is your final notice. Failure to verify your account at {url} "
    "within 24 hours will result in permanent suspension.",

    "Your refund of $850 is ready to be processed. Click {url} now and "
    "provide your account number to receive the funds immediately.",
]

SUSPICIOUS_URLS = [
    "http://192.168.44.12/verify-account",
    "http://secure-paypal-update.tk/login",
    "http://bit.ly/3xJa9Lm",
    "http://appleid-verify-secure.info/reset",
    "http://amaz0n-billing-support.com/confirm",
    "http://tinyurl.com/verify-now-2024",
    "http://bankofamerica-alert.co/secure",
    "http://irs-refund-claim.net/form",
    "http://update-account-now.xyz/login.php",
    "http://netfl1x-billing.com/payment",
]


# --------------------------------------------------------------------------
# Building blocks for LEGITIMATE emails
# --------------------------------------------------------------------------

LEGIT_SUBJECTS = [
    "Team Meeting Rescheduled to 3 PM",
    "Your Order Has Shipped",
    "Invoice #4521 for August Services",
    "Weekly Newsletter - Product Updates",
    "Reminder: Project Deadline Next Friday",
    "Welcome to Our Community!",
    "Your Monthly Statement Is Ready",
    "Notes from Today's Standup",
    "Lunch Plans This Week?",
    "Conference Registration Confirmation",
    "Your Flight Itinerary for Next Week",
    "Updated Company Holiday Schedule",
    "Thanks for Your Recent Purchase",
    "New Blog Post: Tips for Remote Work",
    "Your Subscription Renewal Confirmation",
]

LEGIT_BODIES = [
    "Hi team, just a reminder that our meeting has been moved to 3 PM "
    "today in the main conference room. Please bring your project updates.",

    "Hello, your recent order #{num} has shipped and is expected to "
    "arrive within 3-5 business days. You can track it using the "
    "carrier's website.",

    "Please find attached invoice #{num} for services rendered in "
    "August. Payment is due within 30 days per our standard terms.",

    "Hi everyone, here's this week's newsletter with updates on new "
    "features, upcoming events, and tips from the community.",

    "Just a friendly reminder that the project deliverable is due next "
    "Friday. Let me know if you need any help finishing up your section.",

    "Welcome aboard! We're excited to have you join our community. Feel "
    "free to introduce yourself in the forum whenever you get a chance.",

    "Your monthly account statement for {num} is now available in your "
    "dashboard. Let us know if you have any questions about the charges.",

    "Here are the notes from today's standup: we discussed progress on "
    "the sprint tasks and identified a couple of blockers to follow up on.",

    "Hey, are you free for lunch on Thursday? Thought it'd be nice to "
    "catch up before the quarter wraps up.",

    "Thank you for registering for the conference. Your confirmation "
    "number is {num}. We look forward to seeing you there.",
]


def _fill_template(template, url_pool=None):
    text = template
    if "{url}" in text and url_pool:
        text = text.replace("{url}", random.choice(url_pool))
    if "{num}" in text:
        text = text.replace("{num}", str(random.randint(1000, 9999)))
    return text


def generate_dataset(n_phishing=300, n_legit=300):
    rows = []

    for _ in range(n_phishing):
        subject = random.choice(PHISHING_SUBJECTS)
        body = _fill_template(random.choice(PHISHING_BODIES), SUSPICIOUS_URLS)
        rows.append({"subject": subject, "body": body, "label": 1})

    for _ in range(n_legit):
        subject = random.choice(LEGIT_SUBJECTS)
        body = _fill_template(random.choice(LEGIT_BODIES))
        rows.append({"subject": subject, "body": body, "label": 0})

    df = pd.DataFrame(rows)
    df["text"] = df["subject"] + " " + df["body"]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df[["text", "subject", "body", "label"]]


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("emails_dataset.csv", index=False)
    print(f"Generated {len(df)} emails -> emails_dataset.csv")
    print(df["label"].value_counts())
