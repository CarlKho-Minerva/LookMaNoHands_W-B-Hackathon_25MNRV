#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

# Carl's note: we extended this code from the Daily SDK, which is licensed under the BSD 2-Clause License.
# # The Daily SDK is available at https://github.com/pipecat-ai/pipecat/tree/main

import asyncio
import os
import sys
import json
import time
from typing import Optional, List
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from runner import configure
from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright
import agentql

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.gemini_multimodal_live.gemini import (
    GeminiMultimodalLiveLLMService,
)
from pipecat.transports.services.daily import DailyParams, DailyTransport

# DuckDuckGo URL
URL = "https://duckduckgo.com/"

# The URL of the external or existing browser you wish to connect
WEBSOCKET_URL = "http://localhost:9222"

# Global instances
browser_instance = None
playwright_instance = None

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Rate limiting settings
RATE_LIMIT_DELAY = 1.0  # seconds between API calls
last_api_call = 0


async def wait_for_rate_limit():
    """Ensure we don't exceed API rate limits"""
    global last_api_call
    now = time.time()
    if now - last_api_call < RATE_LIMIT_DELAY:
        await asyncio.sleep(RATE_LIMIT_DELAY - (now - last_api_call))
    last_api_call = time.time()


def perform_search_sync(query: str) -> List[str]:
    """Synchronous function to perform the search using AgentQL with local browser."""
    global browser_instance, playwright_instance

    try:
        # Initialize Playwright if not already done
        if not playwright_instance:
            playwright_instance = sync_playwright().start()
            logger.debug("Started Playwright")

        # Connect to existing browser or create new connection
        if not browser_instance:
            try:
                browser_instance = playwright_instance.chromium.connect_over_cdp(
                    WEBSOCKET_URL
                )
                logger.debug("Connected to local browser")
            except Exception as e:
                logger.error(f"Failed to connect to browser: {str(e)}")
                return [
                    "Failed to connect to browser. Make sure Brave is running with --remote-debugging-port=9222"
                ]

        # Create a new context if none exists
        try:
            context = browser_instance.contexts[0]
        except IndexError:
            context = browser_instance.new_context()
            logger.debug("Created new browser context")

        # Create a new tab in the browser window
        try:
            page = context.new_page()
            page = agentql.wrap(page)
            logger.debug("Created new browser tab")
        except Exception as e:
            logger.error(f"Failed to create new page: {str(e)}")
            return ["Failed to create new browser tab"]

        logger.debug("Navigating to DuckDuckGo...")
        page.goto(URL)
        page.wait_for_timeout(1000)  # Let page load visibly

        # Get and use the search bar
        logger.debug("Looking for search bar...")
        search_bar = page.get_by_prompt("the search bar")
        if not search_bar:
            logger.error("Could not find search bar")
            return ["Could not find search bar"]

        # Fill the search query with visual typing
        logger.debug(f"Typing search query: {query}")
        search_bar.press_sequentially(query)  # Type with visible keystrokes
        page.wait_for_timeout(500)  # Pause for visual effect

        # Try to find and click search button, otherwise press Enter
        logger.debug("Looking for search button...")
        search_button = page.get_by_prompt("the search button")
        if search_button:
            logger.debug("Clicking search button")
            search_button.click()
        else:
            logger.debug("No button found, pressing Enter")
            search_bar.press("Enter")

        # Wait for results to load with visual feedback
        logger.debug("Waiting for results to load...")
        page.wait_for_timeout(2000)

        # Extract search results
        logger.debug("Extracting search results...")
        js_query = (
            "() => Array.from(document.querySelectorAll('.result__title'))"
            ".slice(0, 3).map(el => el.textContent.trim())"
            ".filter(Boolean)"
        )
        results = page.evaluate(js_query)

        # Log results for debugging
        if results:
            logger.debug(f"Found {len(results)} results")
        else:
            logger.debug("No results found")

        return results if results else ["No results found"]

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return [f"Error performing search: {str(e)}"]


# Thread pool for running sync code
thread_pool = ThreadPoolExecutor(max_workers=2)


async def perform_search(query: str) -> List[str]:
    """Async wrapper that runs the sync search in a thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(thread_pool, perform_search_sync, query)


async def start_web_search_function(
    function_name: str,
    tool_call_id: str,
    args: dict,
    llm: GeminiMultimodalLiveLLMService,
    context: Optional[dict],
    result_callback: callable,
) -> None:
    """Execute the initial search function and format results."""
    try:
        search_results = await perform_search(args["query"])
        success = len(search_results) > 0 and search_results[0] != "No results found"
        formatted_results = {
            "results": search_results,
            "query": args["query"],
            "success": success,
            "is_first_search": True,
        }
        await result_callback(formatted_results)
    except Exception as e:
        logger.error(f"Initial search function error: {str(e)}")
        await result_callback(
            {"error": str(e), "query": args.get("query", "unknown"), "success": False}
        )


async def perform_next_search_function(
    function_name: str,
    tool_call_id: str,
    args: dict,
    llm: GeminiMultimodalLiveLLMService,
    context: Optional[dict],
    result_callback: callable,
) -> None:
    """Execute subsequent search function and format results."""
    try:
        search_results = await perform_search(args["query"])
        success = len(search_results) > 0 and search_results[0] != "No results found"
        formatted_results = {
            "results": search_results,
            "query": args["query"],
            "success": success,
            "is_subsequent_search": True,
        }
        await result_callback(formatted_results)
    except Exception as e:
        logger.error(f"Subsequent search function error: {str(e)}")
        await result_callback(
            {"error": str(e), "query": args.get("query", "unknown"), "success": False}
        )


tools = [
    {
        "function_declarations": [
            {
                "name": "start_web_search",
                "description": "Start a new DuckDuckGo search session. Call this for the first search when no browser is open yet.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to type into the search bar",
                        }
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "perform_next_search",
                "description": "Perform another search in the existing browser session. Call this for subsequent searches.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to type into the search bar",
                        }
                    },
                    "required": ["query"],
                },
            },
        ]
    }
]

system_instruction = """
You are a multimodal AI assistant that directly controls a web browser using AgentQL.
You can see the user's screen and actively perform web searches by controlling their browser.

IMPORTANT: For searches, you must use these functions:
1. For the FIRST search in a session, use start_web_search
2. For ALL SUBSEQUENT searches, use perform_next_search

When users ask you to search:
1. Choose the correct function based on whether it's the first or subsequent search
2. While the search is happening, narrate what you observe:
   - The query being typed into the search bar
   - The search button being clicked
   - The results loading on the page
3. After the search completes:
   - Describe the results you see
   - Offer to modify search terms if needed
   - Ask if they want to try a different search

Remember:
- Use start_web_search for the first search only
- Use perform_next_search for all subsequent searches
- Describe what you see on the screen in real-time
- If the search interface isn't visible, ask the user to share their screen
- Guide users through the visual search experience

Example interaction:
User: "Search for dogs"
Assistant: "I'll start a new search for dogs using the browser..."
[Calls start_web_search with query="dogs"]

User: "Now search for cats"
Assistant: "I'll perform another search for cats..."
[Calls perform_next_search with query="cats"]
"""


# Cleanup function to close browser and playwright on exit
def cleanup():
    global browser_instance, playwright_instance
    if browser_instance:
        try:
            browser_instance.close()
            logger.debug("Closed browser connection")
        except Exception as e:
            logger.error(f"Error closing browser: {str(e)}")

    if playwright_instance:
        try:
            playwright_instance.stop()
            logger.debug("Stopped Playwright")
        except Exception as e:
            logger.error(f"Error stopping Playwright: {str(e)}")


# Register cleanup function to run on program exit
import atexit

atexit.register(cleanup)


async def main():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")

    # Variable to store last response
    last_response = None

    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Search Assistant Bot",
            DailyParams(
                audio_out_enabled=True,
                vad_enabled=True,
                vad_audio_passthrough=True,
                vad_analyzer=SileroVADAnalyzer(
                    params=VADParams(
                        stop_secs=0.5,
                        confidence=0.8,  # Better accuracy
                    )
                ),
            ),
        )

        llm = GeminiMultimodalLiveLLMService(
            api_key=google_api_key,
            voice_id="Puck",
            system_instruction=system_instruction,
            transcribe_user_audio=True,
            transcribe_model_audio=True,
            inference_on_context_initialization=False,
            tools=tools,
            temperature=0.7,
            top_p=0.95,
        )

        # Register the search functions
        llm.register_function("start_web_search", start_web_search_function)
        llm.register_function("perform_next_search", perform_next_search_function)

        async def on_model_response(response):
            try:
                logger.debug("=== RESPONSE CONFIRMATION ===")
                logger.debug(f"Response: {response}")
                logger.debug("============================")
            except Exception as e:
                logger.error(f"Error in model response handler: {e}")

        logger.debug("Registering model response callback...")
        llm.on_model_response = on_model_response

        async def on_model_transcription(text):
            try:
                if text:
                    logger.debug(f"Transcribed text: {text}")
            except Exception as e:
                logger.error(f"Error in transcription handler: {e}")

        llm.on_model_transcription = on_model_transcription

        welcome_msg = (
            "Welcome me and tell me you can help me search the web. "
            "Mention that you need to see my screen first by having "
            "me click 'Share Screen' below."
        )
        context = OpenAILLMContext([{"role": "user", "content": welcome_msg}])
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),
                context_aggregator.user(),
                llm,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            # Enable both camera and screenshare
            await transport.capture_participant_video(
                participant["id"], framerate=1, video_source="screenVideo"
            )
            await transport.capture_participant_video(
                participant["id"], framerate=1, video_source="camera"
            )
            await task.queue_frames([context_aggregator.user().get_context_frame()])
            await asyncio.sleep(3)
            logger.debug("Unpausing audio and video")
            llm.set_audio_input_paused(False)
            llm.set_video_input_paused(False)

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
