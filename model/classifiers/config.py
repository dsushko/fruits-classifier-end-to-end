from typing import Optional

from pydantic import BaseModel, Field, validator

class ClassifierConfig(BaseModel):
    name: str = Field()
    params: Optional[dict] = Field(default={})