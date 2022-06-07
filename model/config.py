from model.classifiers.config import ClassifierConfig
from model.preprocessing.config import PreprocessingConfig

from pydantic import BaseModel, Field, validator, root_validator

class ModelConfig(BaseModel):
    classifier: ClassifierConfig = Field()
    preprocessing: PreprocessingConfig = Field()