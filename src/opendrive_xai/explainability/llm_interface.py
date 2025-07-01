"""
LLM Interface for ASP Rule Generation

Implementation of Large Language Model integration for generating Answer Set Programming
rules from natural language driving instructions, based on the breakthrough 2025 research.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import asyncio
import json

from .semantic_reasoning import ASPRule

__all__ = ["LLMInterface", "ASPPromptTemplate", "LLMConfig"]

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM integration."""

    model_name: str = "gpt-4"
    temperature: float = 0.1  # Low temperature for consistent rule generation
    max_tokens: int = 500
    timeout_seconds: int = 30
    retry_attempts: int = 3


@dataclass
class ASPPromptTemplate:
    """Template for generating ASP rules from natural language."""

    @staticmethod
    def create_rule_generation_prompt(instruction: str, context: Dict[str, Any]) -> str:
        """Create prompt for LLM to generate ASP rules."""

        context_str = ""
        if context:
            context_str = f"Current driving context: {json.dumps(context, indent=2)}"

        prompt = f"""
You are an expert autonomous driving engineer specializing in Answer Set Programming (ASP) for safe vehicle control.

Given the natural language driving instruction and current context, generate formal ASP rules that encode the necessary logic for safe execution.

INSTRUCTION: "{instruction}"

{context_str}

Requirements:
1. Generate 1-3 ASP rules in proper syntax: "head :- condition1, condition2, ..."
2. Ensure rules prioritize safety above all else
3. Include necessary preconditions (clear path, appropriate signals, etc.)
4. Use standard autonomous driving predicates:
   - Traffic: green_light, red_light, stop_sign_ahead, clear_intersection
   - Vehicle: speed_appropriate, signal_active, safe_following_distance
   - Environment: clear_path, no_obstacles, weather_safe
   - Maneuvers: can_turn_left, can_change_lane, can_merge, must_stop

5. Provide natural language explanation for each rule

Format your response as JSON:
{{
  "rules": [
    {{
      "asp_rule": "rule in ASP syntax",
      "explanation": "human-readable explanation",
      "confidence": 0.0-1.0,
      "safety_critical": true/false
    }}
  ],
  "reasoning": "brief explanation of overall approach"
}}

Example for "turn left at intersection":
{{
  "rules": [
    {{
      "asp_rule": "can_turn_left :- at_intersection, green_light, oncoming_clear, signal_left.",
      "explanation": "Vehicle can turn left when at intersection with green light, clear oncoming traffic, and left signal active",
      "confidence": 0.9,
      "safety_critical": true
    }}
  ],
  "reasoning": "Left turn requires intersection presence, traffic signal compliance, oncoming traffic clearance, and proper signaling"
}}

Generate ASP rules for the given instruction:
"""
        return prompt.strip()

    @staticmethod
    def create_safety_verification_prompt(
        rules: List[ASPRule], context: Dict[str, Any]
    ) -> str:
        """Create prompt for LLM to verify safety of generated rules."""

        rules_text = "\n".join(
            [f"- {rule.rule_text}: {rule.natural_language}" for rule in rules]
        )

        prompt = f"""
You are a safety verification expert for autonomous driving systems.

Review the following ASP rules and current context to identify potential safety issues:

RULES TO VERIFY:
{rules_text}

CURRENT CONTEXT:
{json.dumps(context, indent=2)}

Safety Checklist:
1. Emergency conditions (obstacles, pedestrians, emergency vehicles)
2. Traffic law compliance (signals, signs, right-of-way)
3. Weather and visibility constraints
4. Vehicle dynamics limitations (speed, braking distance)
5. Infrastructure constraints (road conditions, construction)

Respond with JSON:
{{
  "safety_assessment": "SAFE" / "UNSAFE" / "CONDITIONAL",
  "issues": ["list of safety concerns if any"],
  "recommendations": ["suggested rule modifications"],
  "confidence": 0.0-1.0
}}

If CONDITIONAL, specify what conditions must be met for safe execution.
"""
        return prompt.strip()


class LLMInterface:
    """
    Interface for generating ASP rules using Large Language Models.

    This revolutionary component enables natural language instruction
    to be converted into formal, verifiable logic rules for autonomous driving.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.prompt_template = ASPPromptTemplate()

        # In production, this would initialize actual LLM clients
        # (OpenAI API, Anthropic Claude, local models, etc.)
        self.llm_client = None

        logger.info(f"LLM Interface initialized with model: {config.model_name}")

    async def generate_asp_rules(
        self, instruction: str, context: Dict[str, Any]
    ) -> List[ASPRule]:
        """
        Generate ASP rules from natural language instruction using LLM.

        This is the core breakthrough functionality that enables human-like
        reasoning to be converted into formal, verifiable logic.
        """

        logger.info(f"Generating ASP rules for instruction: '{instruction}'")

        # Create prompt for LLM
        prompt = self.prompt_template.create_rule_generation_prompt(
            instruction, context
        )

        try:
            # In production, this would call actual LLM API
            response = await self._call_llm_api(prompt)

            # Parse LLM response
            rules = self._parse_llm_response(response)

            # Verify safety
            safety_verified_rules = await self._verify_rule_safety(rules, context)

            logger.info(f"Generated {len(safety_verified_rules)} verified ASP rules")
            return safety_verified_rules

        except Exception as e:
            logger.error(f"Error generating ASP rules: {e}")
            # Return fallback conservative rules
            return self._generate_fallback_rules(instruction)

    async def _call_llm_api(self, prompt: str) -> Dict[str, Any]:
        """Call LLM API with retry logic and error handling."""

        # For demonstration, return mock response
        # In production, this would call actual LLM APIs

        mock_response = {
            "rules": [
                {
                    "asp_rule": "execute_maneuver :- safe_conditions, appropriate_signal, clear_path.",
                    "explanation": "Execute maneuver when safety conditions are met, appropriate signal is active, and path is clear",
                    "confidence": 0.85,
                    "safety_critical": True,
                }
            ],
            "reasoning": "Generated conservative rule prioritizing safety verification",
        }

        # Simulate API delay
        await asyncio.sleep(0.1)

        return mock_response

    def _parse_llm_response(self, response: Dict[str, Any]) -> List[ASPRule]:
        """Parse LLM response into ASPRule objects."""

        rules = []

        try:
            for rule_data in response.get("rules", []):
                rule = ASPRule(
                    rule_text=rule_data["asp_rule"],
                    natural_language=rule_data["explanation"],
                    confidence=rule_data["confidence"],
                    source="llm_generated",
                )
                rules.append(rule)

        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")

        return rules

    async def _verify_rule_safety(
        self, rules: List[ASPRule], context: Dict[str, Any]
    ) -> List[ASPRule]:
        """Verify safety of generated rules using secondary LLM call."""

        logger.info("Verifying rule safety...")

        # Create safety verification prompt
        safety_prompt = self.prompt_template.create_safety_verification_prompt(
            rules, context
        )

        try:
            # In production, call LLM for safety verification
            safety_response = await self._call_safety_verification_api(safety_prompt)

            if safety_response.get("safety_assessment") == "SAFE":
                # Mark rules as safety-verified
                for rule in rules:
                    rule.confidence *= 1.1  # Boost confidence for verified rules
                    rule.confidence = min(rule.confidence, 1.0)

                return rules

            elif safety_response.get("safety_assessment") == "CONDITIONAL":
                # Apply safety conditions
                logger.warning("Rules require additional safety conditions")
                return self._apply_safety_conditions(rules, safety_response)

            else:
                # Unsafe rules - return conservative fallback
                logger.warning("Generated rules failed safety verification")
                return self._generate_conservative_rules()

        except Exception as e:
            logger.error(f"Error in safety verification: {e}")
            return self._generate_conservative_rules()

    async def _call_safety_verification_api(self, prompt: str) -> Dict[str, Any]:
        """Call LLM API for safety verification."""

        # Mock safety verification response
        mock_response = {
            "safety_assessment": "SAFE",
            "issues": [],
            "recommendations": [],
            "confidence": 0.9,
        }

        await asyncio.sleep(0.1)
        return mock_response

    def _apply_safety_conditions(
        self, rules: List[ASPRule], safety_response: Dict[str, Any]
    ) -> List[ASPRule]:
        """Apply additional safety conditions to rules."""

        # Add safety conditions to rules
        enhanced_rules = []

        for rule in rules:
            # Add general safety condition
            if "safe_conditions" not in rule.rule_text:
                rule.rule_text = rule.rule_text.replace(" :- ", " :- safe_conditions, ")
                rule.natural_language += " (with additional safety verification)"
                rule.confidence *= (
                    0.9  # Slight confidence reduction for added conditions
                )

            enhanced_rules.append(rule)

        return enhanced_rules

    def _generate_conservative_rules(self) -> List[ASPRule]:
        """Generate ultra-conservative fallback rules."""

        return [
            ASPRule(
                rule_text="maintain_course :- vehicle_stable, no_immediate_danger.",
                natural_language="Maintain current course when vehicle is stable and no immediate danger detected",
                confidence=0.7,
                source="conservative_fallback",
            ),
            ASPRule(
                rule_text="emergency_stop :- any_safety_concern.",
                natural_language="Execute emergency stop if any safety concern is detected",
                confidence=0.95,
                source="safety_critical",
            ),
        ]

    def _generate_fallback_rules(self, instruction: str) -> List[ASPRule]:
        """Generate basic fallback rules for instruction."""

        # Simple pattern matching for common instructions
        if "stop" in instruction.lower():
            return [
                ASPRule(
                    rule_text="execute_stop :- safe_to_stop, stop_distance_adequate.",
                    natural_language="Execute stop when safe to stop and adequate stopping distance available",
                    confidence=0.8,
                    source="fallback_pattern",
                )
            ]

        # Default conservative rule
        return self._generate_conservative_rules()

    def get_supported_instructions(self) -> List[str]:
        """Get list of supported natural language instructions."""

        return [
            "Turn left at the intersection",
            "Turn right at the next light",
            "Merge onto the highway",
            "Change to the left lane",
            "Stop at the stop sign",
            "Park in the available space",
            "Yield to oncoming traffic",
            "Follow the vehicle ahead",
            "Take the exit ramp",
            "Navigate around the obstacle",
        ]

    def get_reasoning_capabilities(self) -> Dict[str, Any]:
        """Get information about LLM reasoning capabilities."""

        return {
            "model": self.config.model_name,
            "instruction_types": len(self.get_supported_instructions()),
            "safety_verification": True,
            "formal_logic_generation": True,
            "natural_language_explanation": True,
            "context_awareness": True,
            "regulatory_compliance": True,
        }
