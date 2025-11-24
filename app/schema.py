from pydantic import BaseModel, Field
from typing import Annotated

class ToxicityRequest(BaseModel):
    text: Annotated[
        str,
        Field(
            ...,
            description="Input text to classify across multiple toxicity categories.",
            min_length=5,
            examples=[
                "You are absolutely terrible at this. Stop talking.",
                "I hope you fail miserably, you're the worst!",
                "I disagree with your opinion, but I respect your perspective.",
            ]
        )
    ]
