from __future__ import annotations

import os
from enum import Enum
from typing import Optional

import instructor
import pytest
from litellm import Router
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator, model_validator


# Enums
class ExperienceLevel(str, Enum):
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"


class CategoryLabel(str, Enum):
    ENGINEERING = "engineering"
    DATA = "data"
    PRODUCT = "product"
    DESIGN = "design"


# Models
class SalaryRange(BaseModel):
    min_amount: Optional[float] = Field(default=None, ge=0)
    max_amount: Optional[float] = Field(default=None, ge=0)
    currency: str = "USD"

    # mode="after" means it will run after the instance is created
    @model_validator(mode="after")
    def validate_range(self) -> "SalaryRange":
        if self.min_amount is not None and self.max_amount is not None:
            if self.min_amount > self.max_amount:
                raise ValueError(
                    f"min_amount ({self.min_amount}) cannot exceed max_amount ({self.max_amount})"
                )
        return self


class JobCategory(BaseModel):
    label: CategoryLabel
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = Field(description="Why this category applies")


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

    # field validator run to validate the individual field of model after model is updated/created
    @field_validator("skills")
    @classmethod
    def deduplicate_skills(cls, v: list[Skill]) -> list[Skill]:
        seen: set[str] = set()
        return [
            s for s in v if not (s.name.lower() in seen or seen.add(s.name.lower()))
        ]

    @model_validator(mode="after")
    def check_confidence_honesty(self) -> "JobPostingAnalysis":
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


# Analyzer
SYSTEM_PROMPT = """You are a job posting analyzer. Extract structured information
from job postings accurately. When a field is not present in the posting, set it
to null and add its name to missing_fields. Be honest with overall_confidence —
only score high when you've extracted most key fields."""

MODEL_LIST = [
    {
        "model_name": "gpt-4o-mini",
        "litellm_params": {
            "model": "openai/gpt-4o-mini",
            "api_key": os.getenv("OPENROUTER_KEY"),
            "base_url": "https://openrouter.ai/api/v1",
            "rpm": 6,
        },
    },
]


class JobAnalyzer:
    def __init__(self, model: str = "gpt-4o-mini"):
        router = Router(model_list=MODEL_LIST)
        self.client = instructor.from_litellm(router.completion)
        self.model = model

    def analyze(self, posting: str) -> dict:
        try:
            data = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_model=JobPostingAnalysis,
                max_retries=3,  # retries on Pydantic validation errors automatically
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": posting},
                ],
            )
            return {"success": True, "data": data}
        except Exception as e:
            return {"success": False, "error": str(e)}


# Unit Tests (no API key required)
def test_salary_validation_triggers_retry():
    with pytest.raises(ValueError, match="cannot exceed"):
        SalaryRange(min_amount=200_000, max_amount=100_000)


def test_confidence_honesty_check():
    with pytest.raises(ValueError, match="too high"):
        JobPostingAnalysis(overall_confidence=0.95)


def test_skill_deduplication():
    analysis = JobPostingAnalysis(
        overall_confidence=0.5,
        skills=[
            Skill(name="Python", is_required=True),
            Skill(name="python", is_required=False),
        ],
    )
    assert len(analysis.skills) == 1


def test_valid_salary_range():
    s = SalaryRange(min_amount=80_000, max_amount=120_000)
    assert s.min_amount < s.max_amount


def test_required_skills_property():
    analysis = JobPostingAnalysis(
        overall_confidence=0.5,
        skills=[
            Skill(name="Python", is_required=True),
            Skill(name="Excel", is_required=False),
        ],
    )
    assert analysis.required_skills == ["Python"]


# Integration Tests (needs OPENAI_API_KEY)
@pytest.mark.integration
class TestAnalyzerIntegration:
    @pytest.fixture
    def analyzer(self):
        return JobAnalyzer()

    def test_complete_extraction(self, analyzer):
        posting = """
        Senior Software Engineer at facebook
        Salary: $150,000 - $200,000
        Requirements: Python (required), Kubernetes (required), AWS (nice-to-have)
        """
        result = analyzer.analyze(posting)
        assert result["success"], result.get("error")
        assert result["data"].experience_level == ExperienceLevel.SENIOR
        assert result["data"].overall_confidence >= 0.7

    def test_partial_extraction(self, analyzer):
        result = analyzer.analyze("Python developer needed.")
        assert result["success"], result.get("error")
        assert len(result["data"].missing_fields) > 0


# manual smoke test
if __name__ == "__main__":
    analyzer = JobAnalyzer()
    sample = """
    Senior Data Engineer at amazon
    Salary: $130,000 - $160,000 USD
    Must have: Python, Spark, Airflow
    Nice to have: dbt, Kafka
    """
    result = analyzer.analyze(sample)
    if result["success"]:
        data = result["data"]
        print(f"Title       : {data.job_title}")
        print(f"Level       : {data.experience_level}")
        print(f"Salary      : {data.salary}")
        print(f"Required    : {data.required_skills}")
        print(f"Confidence  : {data.overall_confidence}")
        print(f"Missing     : {data.missing_fields}")
    else:
        print("Error:", result["error"])
