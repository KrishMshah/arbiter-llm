"""
scripts/test_cost_fix.py — throwaway sanity check, not part of the pipeline.
Confirms create_with_completion() + real usage capture actually works
before trusting it for Arena spend. Delete this file once confirmed.
"""

from src.critics import CriticA, CriticB
from src.pipeline import cost_from_tokens

TEST_TEXT = "The capital of France is Paris, and the Eiffel Tower was completed in 1889."
DIMENSIONS = ["factual_accuracy", "completeness"]

for critic_cls in [CriticA, CriticB]:
    critic = critic_cls()
    critique = critic.evaluate(TEST_TEXT, DIMENSIONS)
    cost = cost_from_tokens(critic.model_used, critique.input_tokens, critique.output_tokens)
    print(f"{critic.critic_id} ({critic.model_used}):")
    print(f"  input_tokens={critique.input_tokens}  output_tokens={critique.output_tokens}")
    print(f"  cost=${cost:.6f}  critic_failed={critique.critic_failed}")
    print()