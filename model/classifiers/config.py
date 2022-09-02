from typing import List, Optional

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

class KerasNetworkParamGridConfig(BaseModel):
    image_size: List[int] = Field()
    batch_size: List[int] = Field()
    loss: List[str] = Field()
    optimizer: List[str] = Field()
    epochs: List[int] = Field()
    enable_early_stopping: Optional[List[bool]] = Field(default=False)