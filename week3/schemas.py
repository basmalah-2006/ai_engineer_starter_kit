from typing import Literal
from pydantic import BaseModel, Field


class EmailTriage(BaseModel):
    intent: Literal["support","sales","billing","general","spam",]

    urgency: Literal["low","medium","high",]

    summary: str = Field(min_length=3, max_length=300)