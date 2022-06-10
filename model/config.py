from model.classifiers.config import ClassifierConfig
from model.preprocessing.config import PreprocessingConfig

from typing import Optional

from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    classifier: ClassifierConfig = Field()
    preprocessing: PreprocessingConfig = Field()
    encode_labels: Optional[bool] = Field(default=False)