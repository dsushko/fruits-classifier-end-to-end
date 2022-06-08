from typing import List, Optional

from pydantic import BaseModel, Field, validator

class PreprocessingParamsConfig(BaseModel):
    max_brightness: int = Field(default=255)
    resize_value: int = Field(default=128)

class PreprocessingConfig(BaseModel):
    unification_steps: List[str] = Field()
    processing_steps: Optional[List[str]] = Field(default=[])
    params: Optional[PreprocessingParamsConfig] = \
        Field(default=PreprocessingParamsConfig())

    @validator('steps')
    def check_presence(cls, v):
        if v is None:
            raise ValueError(f'{v} is not defined in cfg')
        return v