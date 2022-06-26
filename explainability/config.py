from pydantic import BaseModel, Field

class LimeExplainabilityConfig(BaseModel):
    num_features: int = Field(default=100)
    hide_rest: bool = Field(default=True)
    positive_only: bool = Field(default=False)
    min_weight: bool = Field(default=0.1)