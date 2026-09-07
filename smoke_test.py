from src.critics import CriticC

result = CriticC().evaluate(
    "The mitochondria is the powerhouse of the cell, and it was discovered by Einstein in 1922.",
    ["factual_accuracy", "completeness"],
)
print(result.model_dump())