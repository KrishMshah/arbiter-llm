from src.critics import CriticA, CriticB, CriticC

text = "The mitochondria is the powerhouse of the cell, and it was discovered by Einstein in 1922."
dims = ["factual_accuracy", "completeness"]

for critic in [CriticA(), CriticB(), CriticC()]:
    result = critic.evaluate(text, dims)
    print(f"\n{critic.critic_id} ({critic.model_used})")
    print(result.model_dump())