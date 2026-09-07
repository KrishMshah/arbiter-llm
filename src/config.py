"""
src/config.py

Central place for env-driven settings. Phase 1 only needs MOCK_MODE and
the ML confidence threshold — API keys are read but unused until Phase 2
swaps critics.py over to real calls.
"""

import os
from dotenv import load_dotenv # type: ignore

load_dotenv()

MOCK_MODE: bool = os.getenv("MOCK_MODE", "true").lower() == "true"

# Per-critic overrides — each defaults to MOCK_MODE's value unless
# individually set in .env. This is what lets Critic C go real (local,
# free) while Critic A/B stay mocked, or all three flip independently.
MOCK_CRITIC_A: bool = os.getenv("MOCK_CRITIC_A", str(MOCK_MODE)).lower() == "true"
MOCK_CRITIC_B: bool = os.getenv("MOCK_CRITIC_B", str(MOCK_MODE)).lower() == "true"
MOCK_CRITIC_C: bool = os.getenv("MOCK_CRITIC_C", str(MOCK_MODE)).lower() == "true"

# API Keys — unused while the corresponding MOCK_CRITIC flag is true.
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Used by arbitrator.py's escalation logic (stub in Phase 1, real in Phase 2)
ML_CONFIDENCE_THRESHOLD: float = float(os.getenv("ML_CONFIDENCE_THRESHOLD", "0.75"))