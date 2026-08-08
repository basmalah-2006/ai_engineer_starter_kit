from dotenv import load_dotenv

load_dotenv()

from service import triage_email


if __name__ == "__main__":
    email_text = """
    Hello,

    I tried to pay for my subscription but the payment failed.
    I need this fixed urgently because my team cannot access the dashboard.

    Thanks,
    Ahmed
    """

    result = triage_email(email_text)

    print(result.model_dump_json(indent=2))