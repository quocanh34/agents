from __future__ import annotations

from typing import AsyncIterable

from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm, tokenize
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import cartesia, deepgram, openai, silero

load_dotenv()

async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
            "You should use short and concise responses, and avoiding usage of unpronouncable punctuation."
        ),
    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    def _estimate_audio_length(text: str, phoneme_dur: float = 0.07) -> float:
        """
        Estimate TTS audio duration using phoneme-based analysis.
        
        Args:
            text (str): Input text to convert to speech
            phoneme_dur (float): Average phoneme duration in seconds. Default 0.07.
        
        Returns:
            float: Estimated audio duration in seconds
        """
        from g2p_en import G2p
        g2p = G2p()
        phonemes = g2p(text)
        return len(phonemes) * phoneme_dur

    def _before_tts_cb(agent: VoicePipelineAgent, text: str | AsyncIterable[str]) -> tuple[str | AsyncIterable[str], float]:
        # Estimate the duration
        estimated_audio_length = _estimate_audio_length(text) if isinstance(text, str) else 0.0
        
        return estimated_audio_length

    # also for this example, we also intensify the keyword "LiveKit" to make it more likely to be
    # recognized with the STT
    deepgram_stt = deepgram.STT(keywords=[("LiveKit", 3.5)])

    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=deepgram_stt,
        llm=openai.LLM(),
        tts=cartesia.TTS(),
        chat_ctx=initial_ctx,
        before_tts_cb=_before_tts_cb,
    )
    agent.start(ctx.room)

    await agent.say("Hey, LiveKit is awesome!", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
