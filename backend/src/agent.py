# Cleaned Wellness Agent (Male voice, slow-speaking)
# Original uploaded file path: /mnt/data/agent.py
# This file reproduces the same functionality as the user's agent.py but
# - removes tutorial/YouTube references and prints
# - uses a male, slower-speaking TTS voice
# - keeps the same tools, JSON persistence, and behaviour

import logging
import json
import os
from datetime import datetime
from typing import Annotated
from dataclasses import dataclass, field, asdict

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    MetricsCollectedEvent,
    RunContext,
    function_tool,
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("wellness_agent")
load_dotenv(".env.local")

# -----------------------------
# Persistence
# -----------------------------
WELLNESS_LOG_FILE = "wellness_log.json"

def get_log_path():
    base_dir = os.path.dirname(__file__)
    backend_dir = os.path.abspath(os.path.join(base_dir, ".."))
    return os.path.join(backend_dir, WELLNESS_LOG_FILE)


def load_history() -> list:
    path = get_log_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding='utf-8') as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning(f"Could not load history: {e}")
        return []


def save_checkin_entry(entry: "CheckInState") -> None:
    path = get_log_path()
    history = load_history()

    record = {
        "timestamp": datetime.now().isoformat(),
        "mood": entry.mood,
        "energy": entry.energy,
        "objectives": entry.objectives,
        "summary": entry.advice_given,
    }

    history.append(record)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding='utf-8') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

    logger.info(f"CHECK-IN SAVED TO {path}")

# -----------------------------
# State
# -----------------------------
@dataclass
class CheckInState:
    mood: str | None = None
    energy: str | None = None
    objectives: list[str] = field(default_factory=list)
    advice_given: str | None = None

    def is_complete(self) -> bool:
        return all([
            self.mood is not None,
            self.energy is not None,
            len(self.objectives) > 0,
        ])

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Userdata:
    current_checkin: CheckInState
    history_summary: str
    session_start: datetime = field(default_factory=datetime.now)

# -----------------------------
# Tools
# -----------------------------
@function_tool
async def record_mood_and_energy(
    ctx: RunContext[Userdata],
    mood: Annotated[str, Field(description="The user's emotional state (e.g., happy, stressed)")],
    energy: Annotated[str, Field(description="The user's energy level (e.g., high, low)")],
) -> str:
    ctx.userdata.current_checkin.mood = mood
    ctx.userdata.current_checkin.energy = energy
    logger.info(f"MOOD LOGGED: {mood} | ENERGY: {energy}")
    return f"Noted: you are feeling {mood} with {energy} energy."


@function_tool
async def record_objectives(
    ctx: RunContext[Userdata],
    objectives: Annotated[list[str], Field(description="List of 1-3 specific goals the user wants to achieve today")],
) -> str:
    ctx.userdata.current_checkin.objectives = objectives
    logger.info(f"OBJECTIVES LOGGED: {objectives}")
    return "Got it — your goals for today are saved."


@function_tool
async def complete_checkin(
    ctx: RunContext[Userdata],
    final_advice_summary: Annotated[str, Field(description="A brief 1-sentence summary of the advice given")],
) -> str:
    state = ctx.userdata.current_checkin
    state.advice_given = final_advice_summary

    if not state.is_complete():
        return "I can't finish yet — I still need your mood, energy, or at least one goal."

    save_checkin_entry(state)

    recap = (
        f"Here is your recap for today:\n"
        f"You are feeling {state.mood} and your energy is {state.energy}.\n"
        f"Your main goals are: {', '.join(state.objectives)}.\n\n"
        f"Remember: {final_advice_summary}\n\n"
        f"I've saved this in your wellness log. Have a wonderful day!"
    )
    logger.info("WELLNESS CHECK-IN COMPLETED")
    return recap

# -----------------------------
# Agent
# -----------------------------
class WellnessAgent(Agent):
    def __init__(self, history_context: str):
        super().__init__(
            instructions=f"""
You are a compassionate, supportive Daily Wellness Companion.

CONTEXT FROM PREVIOUS SESSIONS:
{history_context}

GOALS FOR THIS SESSION:
1. Ask how they are feeling (mood) and their energy level.
2. Ask for 1-3 simple objectives for the day.
3. Offer small, grounded, NON-MEDICAL advice.
4. Summarize and save the check-in using the provided tools.

SAFETY:
- You are NOT a doctor or therapist.
- Do NOT diagnose or prescribe.
- If user mentions self-harm or crisis, suggest professional help immediately.

Use the available tools to record data as the user speaks.
""",
            tools=[record_mood_and_energy, record_objectives, complete_checkin],
        )

# -----------------------------
# Entrypoint
# -----------------------------

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # Load history
    history = load_history()
    history_summary = "No previous history found. This is the first session."

    if history:
        last_entry = history[-1]
        history_summary = (
            f"Last check-in was on {last_entry.get('timestamp', 'unknown date')}. "
            f"User felt {last_entry.get('mood')} with {last_entry.get('energy')} energy. "
            f"Their goals were: {', '.join(last_entry.get('objectives', []))}."
        )
        logger.info("HISTORY LOADED: %s", history_summary)
    else:
        logger.info("NO HISTORY FOUND")

    userdata = Userdata(
        current_checkin=CheckInState(),
        history_summary=history_summary,
    )

    # Agent session: use male voice with slower pacing
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",  # male voice
            style="Conversation",
            text_pacing=True,       # encourage measured/clear pacing
            # If the TTS provider supports a rate parameter, it can be set here (optional/depends on plugin)
            # e.g. speaking_rate=0.85
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=WellnessAgent(history_context=history_summary),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
