# Multimodal Search Assistant 🔍 🎙️ 🤖

A powerful AI-powered search assistant that combines voice interaction, screen sharing, and live browser control to provide an interactive and intuitive search experience.

![(2) Look ma no Hands  Voice → Browser Control 00-01-55](https://github.com/user-attachments/assets/00ce67a2-05cd-484e-b335-e492b52b60fe)
[Demo Video](https://www.youtube.com/watch?v=j96sa4azdtM)

## Features

### 1. Interactive Visual Search

- **Live Browser Control**: Watch as the AI controls your browser in real-time
- **Visual Feedback**: See searches happen right before your eyes
- **DuckDuckGo Integration**: Privacy-focused search engine integration

### 2. Voice Interaction

- **Natural Conversations**: Talk to the assistant naturally
- **Real-time Transcription**: Your voice is converted to text instantly
- **AI Voice Responses**: Get spoken responses using high-quality TTS

### 3. Screen Sharing

- **Live Screen Viewing**: AI can see your screen and guide you
- **Visual Context**: Assistant understands what you're looking at
- **Interactive Guidance**: Get real-time visual feedback

## Quick Start

1. **Environment Setup**

   ```bash
   # Clone the repository
   git clone [your-repo-url]
   cd multimodal-video-bot

   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Configure Browser**

   ```bash
   # Launch Brave browser with remote debugging
   /Applications/Brave\ Browser.app/Contents/MacOS/Brave\ Browser --remote-debugging-port=9222
   ```

3. **Set Environment Variables**
   Create a `.env` file with:

   ```
   GOOGLE_API_KEY=your_gemini_api_key
   ```

4. **Run the Assistant**

   ```bash
   python main.py
   ```

5. **Start Interacting**
   - Click "Share Screen" when prompted
   - Start talking to the assistant
   - Watch as it performs searches in real-time

## How It Works

1. **Voice Input**
   - Your voice is captured and transcribed in real-time
   - Natural language processing understands your intent

2. **Visual Processing**
   - AI monitors your shared screen
   - Understands the browser state and search context

3. **Browser Control**
   - Uses AgentQL to control the browser programmatically
   - Performs searches with visual feedback
   - Extracts and summarizes search results

4. **Intelligent Response**
   - Combines search results with context
   - Provides spoken and visual feedback
   - Offers follow-up suggestions

## Technologies Used

- **Pipecat SDK**: Core framework for multimodal AI
- **Gemini Pro**: Advanced language model for understanding and generation
- **AgentQL**: Browser automation and control
- **Playwright**: Web browser instrumentation
- **DuckDuckGo**: Privacy-focused search engine
- **Daily.co**: Real-time video and audio communication

## Demo Tips

1. **Start with Simple Searches**
   - "Search for recent tech news"
   - "Show me pictures of puppies"
   - "Find recipes for pasta"

2. **Try Follow-up Queries**
   - "Now search for related topics"
   - "Can you modify the search terms?"
   - "Show me more specific results"

3. **Explore Visual Features**
   - Watch the search happen live
   - Observe the AI's browser control
   - See results appear in real-time

## Limitations & Known Issues

- Requires Brave browser with remote debugging enabled
- Screen sharing must be enabled for visual features
- May need to restart browser if connection is lost
- Rate limits apply to API calls

## Future Improvements

- [ ] Support for multiple browser types
- [ ] Clicking Support via AgentQL
- [ ] Enhanced error recovery
- [ ] More search engine integrations
- [ ] Improved visual result parsing
- [ ] Better context retention between searches

## Contributing

This project was created for the Multimodal AI Agents - 2 day Hackathon. Feel free to fork and improve!

## License

BSD 2-Clause License - See LICENSE file for details
