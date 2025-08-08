# NPCrafter-Backend: Context-Aware, Talking NPC Generator for Games

## Overview

NPCrafter is a powerful FastAPI-based tool designed for game developers to create dynamic, context-aware, and voice-enabled non-player characters (NPCs) for games. It leverages advanced AI models, including Groq for dialogue generation and OpenAI's Text-to-Speech (TTS) for voice synthesis, to produce immersive NPCs that react to player states, align with game contexts, and are fully customizable in personality, style, and emotion.

### Key Features

- **Dynamic NPC Generation**: Create NPCs with tailored personalities, backstories, and dialogue goals based on game context and player stats.
- **Persona Matching**: Automatically selects the most suitable persona (e.g., `Drakblade`, `Gloam`, `Steelord`) from a predefined list based on generated NPC traits using Groq.
- **Context-Aware Dialogue**: NPCs respond dynamically to player attributes like health, inventory, completed quests, and location.
- **Voice Synthesis**: Generate realistic voice samples using OpenAI's TTS API, with customizable tones and emotions.
- **Customizable Styles**: Support for various visual (e.g., realistic, anime, pixel-art) and voice styles to match game aesthetics.
- **Game Data Integration**: Upload game state files to provide additional context for NPC behavior.
- **Exportable Assets**: Export NPC assets (dialogue, visuals, audio, and 3D models) for seamless integration into game engines.

## Getting Started

### Prerequisites

- **Python**: 3.8 or higher
- **pip**: Python package manager
- **Environment Variables**:
  - `GROQ_API_KEY`: For Groq-based dialogue generation
  - `OPENAI_API_KEY`: For OpenAI TTS voice synthesis
- **Dependencies**: Listed in `requirements.txt`

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sparsh-karna/NPCrafter.git
   cd NPCrafter
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv myEnv
   ```

3. **Activate the Virtual Environment**:
   - On macOS/Linux:
     ```bash
     source myEnv/bin/activate
     ```
   - On Windows:
     ```bash
     myEnv\Scripts\activate
     ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Set Up Environment Variables**:
   Create a `.env` file in the project root and add your API keys:
   ```env
   GROQ_API_KEY=your_groq_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

### Running the API

Start the FastAPI server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

The API will be available at `http://localhost:8001`.

### API Documentation

Access interactive API documentation:
- **Swagger UI**: `http://localhost:8001/docs`
- **ReDoc**: `http://localhost:8001/redoc`

## Key Components

### Models

- **`NPCTraits`**: Defines NPC characteristics, including name, personality traits, dialogue goal, backstory, visual style, voice style, and gender.
- **`PlayerStats`**: Captures player-related data such as health, inventory, completed quests, battle logs, achievements, and location.
- **`NPCDialogue`**: Represents NPC dialogue with text, emotion, context references, audio URL, and optional audio content.
- **`PersonaCreationResponse`**: Returns comprehensive NPC details, including a `selected_persona` field for the best-matching persona from a predefined list.

### Personas

NPCrafter supports a curated list of personas for visual and thematic alignment:
- **Barkel**: Low-poly cartoon dog
- **Boneguard**: Armored undead skeleton warrior
- **Bricktee**: Blocky toy figure with red shirt
- **Drakblade**: Armored dragon character with a sword
- **Gloam**: Hooded robed figure with glowing eyes
- **Pawvo**: Cartoon dog character waving
- **Plumeca**: Cartoon Aztec child with feathered headdress
- **Popestep**: Cartoon pope with sunglasses and sneakers
- **Steelord**: Cartoon armored knight with sword

The API uses Groq to match NPC personality traits to these personas, ensuring thematic consistency.

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Returns a welcome message and API version. |
| `/external-api-status` | GET | Checks the availability of the external API. |
| `/generate-npc` | POST | Generates an NPC based on provided parameters. |
| `/random-npc` | POST | Creates a random NPC with minimal input. |
| `/interact-with-npc` | POST | Interacts with a previously generated NPC using player input. |
| `/preview-dialogue` | POST | Generates a preview of NPC dialogue based on traits and player context. |
| `/upload-game-data` | POST | Uploads game data files (e.g., JSON state files) for NPC context. |
| `/export-npc/{npc_id}` | GET | Exports NPC assets (visuals, audio, models, dialogue, or traits). |
| `/npc/{npc_id}` | GET | Retrieves an NPC by its ID. |
| `/save-npc` | POST | Saves or updates an NPC in the database. |
| `/voice-styles` | GET | Lists available voice styles for NPC generation. |
| `/visual-styles` | GET | Lists available visual styles (e.g., anime, realistic). |
| `/npc/create-persona` | POST | Creates a detailed NPC persona with voice samples and context-aware features. |
| `/npc/create-persona-langchain` | POST | Creates an NPC persona using LangChain and Groq, including persona selection. |
| `/npc/speak` | POST | Generates NPC speech audio using OpenAI's TTS API. |
| `/audio/{filename}` | GET | Serves generated audio files. |
| `/npc/voice-options` | GET | Returns available OpenAI voice options and emotions. |

### Example Usage

#### Request to `/npc/create-persona-langchain`

**Request Body**:
```json
{
  "game_context": "A medieval fantasy world filled with dragons, knights, and ancient magic",
  "npc_context": "A cryptic warrior who guards an ancient relic in a hidden temple",
  "gender": "male"
}
```

**Response**:
```json
{
  "npc_name": "Valthor the Veiled",
  "personality_traits": [
    {"trait": "mysterious", "intensity": 0.9, "description": "Speaks in riddles"},
    {"trait": "loyal", "intensity": 0.8, "description": "Devoted to protecting the relic"},
    {"trait": "stoic", "intensity": 0.7, "description": "Calm under pressure"},
    {"trait": "wise", "intensity": 0.6, "description": "Knowledgeable in ancient lore"}
  ],
  "dialogue_goal": "Guide worthy adventurers to the relic",
  "backstory": "Valthor was once a knight of the Dragon Order, now guarding a powerful relic in solitude.",
  "visual_style": "realistic",
  "voice_style": {
    "tone": "deep",
    "accent": "neutral",
    "pace": "measured",
    "gender": "male"
  },
  "sample_dialogue": [
    "Only the worthy may approach the relic. What drives you, traveler?",
    "The temple's secrets are not easily won. Prove your resolve.",
    "I have guarded this place for centuries, and I shall remain."
  ],
  "intents": ["guidance", "quest-giving", "testing", "storytelling"],
  "memory_id": "npc-5678",
  "player_context": {},
  "selected_persona": "Gloam"
}
```

## Current Implementation Notes

- **Persona Selection**: The `/npc/create-persona-langchain` endpoint uses Groq to match NPC personality traits to a predefined persona list, enhancing visual and thematic consistency.
- **Dialogue Generation**: Powered by Groq's `llama3-8b-8192` model, ensuring context-aware and immersive NPC interactions.
- **Voice Synthesis**: Utilizes OpenAI's TTS API with customizable voices (e.g., `alloy`, `sage`, `onyx`) and emotions (e.g., happy, mysterious, stern).
- **Dummy Outputs**: Some endpoints (e.g., `/generate-npc`, `/interact-with-npc`) return placeholder data for demonstration purposes. In a production environment, these would integrate with a database and external APIs.

## Future Enhancements

- **Advanced AI Integration**: Incorporate GPT-4o for more nuanced dialogue generation.
- **Image Generation**: Add support for generating NPC visuals using AI-based image generators.
- **3D Model Conversion**: Integrate tools like HunYuan for converting NPC designs into 3D models.
- **Enhanced Voice Options**: Support additional TTS providers (e.g., ElevenLabs, Bark) for diverse voice styles.
- **Lip Sync**: Implement Wav2Lip for synchronized lip movements in NPC animations.
- **User Authentication**: Add secure user authentication and project-saving capabilities.
- **Database Integration**: Replace dummy data with a persistent database for NPC storage and retrieval.
- **Real-Time Interaction**: Enable real-time NPC interactions with websocket support.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please ensure your code follows the project's style guidelines and includes appropriate tests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For questions or feedback, contact the project maintainer at [your-email@example.com] or open an issue on GitHub.
