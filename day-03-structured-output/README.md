# Day 3: Validation & Testing LLM Outputs

The moment we interact with LLMs, we get probabilistic output. They'll return `"price": "$45.99"` one time and `"price": 45.99` the next. Sometimes, they even forget required fields. This might not look like a big deal in a general chatbot, but in production, these inconsistencies crash pipelines and corrupt data. That's why we need validation to make sure LLM outputs are uniform and clean.

## 1. Establishing Boundaries with Enums

First of all, the simplest way to make LLM outputs strict is with enums. Enums are used when we want the LLM to pick from specific options:

```python
from enum import Enum

class ExperienceLevel(str, Enum):
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
```

Now the LLM returns only these exact values. Nothing like "Senior level" or "Sr."

## 2. Managing Complexity with Nested Pydantic Models

But real data is more complex than single values. So, when we work with nested structures, we need nested models and validation functions. For example, a salary should always be between a minimum and maximum point:

```python
from pydantic import BaseModel, Field, model_validator
from typing import Optional

class SalaryRange(BaseModel):
    min_amount: Optional[float] = Field(default=None, ge=0)
    max_amount: Optional[float] = Field(default=None, ge=0)
    currency: str = "USD"

    @model_validator(mode="after")
    def validate_range(self) -> "SalaryRange":
        if self.min_amount is not None and self.max_amount is not None:
            if self.min_amount > self.max_amount:
                raise ValueError(
                    f"min_amount ({self.min_amount}) cannot exceed max_amount ({self.max_amount})"
                )
        return self


class JobPosting(BaseModel):
    job_title: str
    company_name: str
    experience_level: Optional[ExperienceLevel] = None
    salary: Optional[SalaryRange] = None
```

Here we are using the `SalaryRange` model inside the `JobPosting` model, and `SalaryRange` has its own rules. If those rules fail, the validation also fails, and we can queue this for retry.

## 3. High-Fidelity Classification and Evidence Tracking

We can also follow the same pattern in classification tasks. Here, we combine enums with confidence scores and evidence to get multi-label classification:

```python
class CategoryLabel(str, Enum):
    ENGINEERING = "engineering"
    DATA = "data"
    PRODUCT = "product"
    DESIGN = "design"

class JobCategory(BaseModel):
    label: CategoryLabel
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = Field(description="Why this category applies")
```
## 4. Advanced Validation: Deduplication and Honesty Checks

Things don't stop here. Real-world data is messy, so we should support partial extraction by making fields optional. On top of that, LLMs seem to be overconfident without having the evidence. So, we should also add validators to catch overconfident LLMs:

```python
class Skill(BaseModel):
    name: str
    is_required: bool = Field(description="True if required, False if nice-to-have")


class JobPostingAnalysis(BaseModel):
    job_title: Optional[str] = None
    company_name: Optional[str] = None
    experience_level: Optional[ExperienceLevel] = None
    salary: Optional[SalaryRange] = None
    
    skills: list[Skill] = Field(default_factory=list)
    categories: list[JobCategory] = Field(default_factory=list)
    
    overall_confidence: float = Field(ge=0.0, le=1.0)
    missing_fields: list[str] = Field(default_factory=list)
    
    @field_validator('skills')
    @classmethod
    def deduplicate_skills(cls, v: list[Skill]) -> list[Skill]:
        seen = set()
        return [s for s in v if not (s.name.lower() in seen or seen.add(s.name.lower()))]
    
    @model_validator(mode='after')
    def check_confidence_honesty(self) -> 'JobPostingAnalysis':
        key_fields = [self.job_title, self.experience_level]
        filled = sum(1 for f in key_fields if f is not None)
        completeness = filled / len(key_fields)
        
        if self.overall_confidence > 0.8 and completeness < 0.5:
            raise ValueError(
                f"Confidence {self.overall_confidence} too high when only "
                f"{filled}/{len(key_fields)} key fields extracted"
            )
        return self
    
    @property
    def required_skills(self) -> list[str]:
        return [s.name for s in self.skills if s.is_required]
```

`Skill` model is partial data here, which the LLM should parse if available. Moreover, we also added a `deduplicate_skills` function so that we have a unique list of skills only. Besides this, the `check_confidence_honesty` validator is useful if the LLM claims 95% confidence but only extracted 1 of 2 key fields.

## 5. Unit Testing: Validating Logic Without API Costs

Now that we have our models, we need to test them. Let's first do unit testing on our models directly, which doesn't need an API key:

```python
import pytest

def test_salary_validation_triggers_retry():
    with pytest.raises(ValueError, match="cannot exceed"):
        SalaryRange(min_amount=200000, max_amount=100000)

def test_confidence_honesty_check():
    with pytest.raises(ValueError, match="too high"):
        JobPostingAnalysis(overall_confidence=0.95)

def test_skill_deduplication():
    analysis = JobPostingAnalysis(
        overall_confidence=0.5,
        skills=[
            Skill(name="Python", is_required=True),
            Skill(name="python", is_required=False),
        ]
    )
    assert len(analysis.skills) == 1
```

## 6. Integration Testing: Verifying Real-World Extraction

After we are confident with our models, it's time to do integration testing with the LLM to test actual extraction:

```python
@pytest.mark.integration
class TestAnalyzerIntegration:
    @pytest.fixture
    def analyzer(self):
        return JobAnalyzer()
    
    def test_complete_extraction(self, analyzer):
        posting = """
        Senior Software Engineer at TechCorp
        Salary: $150,000 - $200,000
        Requirements: Python, Kubernetes, AWS
        """
        result = analyzer.analyze(posting)
        
        assert result["success"]
        assert result["data"].experience_level == ExperienceLevel.SENIOR
        assert result["data"].overall_confidence >= 0.7
    
    def test_partial_extraction(self, analyzer):
        result = analyzer.analyze("Python developer needed.")
        assert result["success"]
        assert len(result["data"].missing_fields) > 0
```

Alright, now we can be confident in our models and LLM output. You should always run unit tests in CI and integration tests before deployment, as they cost LLM tokens.