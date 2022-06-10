from typing import Optional

from pydantic import BaseModel, Field

class ClassifierConfig(BaseModel):
    name: str = Field()
    params: Optional[dict] = Field(default={})

class KerasNetworkConfig(BaseModel):
    image_size: int = Field()
    batch_size: int = Field()
    loss: str = Field()
    optimizer: str = Field()
    epochs: int = Field()
    enable_early_stopping: Optional[bool] = Field(default=False)