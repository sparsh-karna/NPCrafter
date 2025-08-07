# NPCrafter: Context-Aware, Talking NPC Generator for Games

## Overview

NPCrafter is a tool that allows game developers to create dynamic, talking NPCs that:
1. React to the player's current state (health, logs, achievements, etc.)
2. Communicate with both text and voice
3. Are fully customizable in style, personality, and emotion
4. Can be previewed and downloaded as complete game-ready assets

## Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone this repository
```
git clone https://github.com/sparsh-karna/NPCrafter.git
cd NPCrafter
```

2. Create a virtual environment
```
python -m venv myEnv
```

3. Activate the virtual environment
```
# On macOS/Linux
source myEnv/bin/activate

# On Windows
myEnv\Scripts\activate
```

4. Install the dependencies
```
pip install -r requirements.txt
```

### Running the API

Start the FastAPI server:
```
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000

### API Documentation

Once the server is running, you can access:
- Interactive API documentation: http://localhost:8000/docs
- Alternative API documentation: http://localhost:8000/redoc

## Current Implementation

This is a minimal implementation with dummy outputs to demonstrate the API structure. 

### Key Models

- `NPCTraits` - Combined model for all NPC characteristics (name, personality traits, backstory, etc.)
- `PlayerStats` - Single model for all player-related information (health, inventory, quests, etc.)
- `NPCDialogue` - Model for dialogue text, emotion and audio

### Key Endpoints

- `POST /generate-npc` - Generate an NPC based on provided parameters
- `POST /random-npc` - Create a random NPC with minimal input
- `POST /interact-with-npc` - Interact with a previously generated NPC
- `POST /preview-dialogue` - Generate a preview of NPC dialogue
- `POST /upload-game-data` - Upload game data files for NPC context
- `GET /export-npc/{npc_id}` - Export generated NPC assets
- `GET /npc/{npc_id}` - Retrieve an NPC by ID
- `POST /save-npc` - Save or update an NPC
- `GET /voice-styles` - Get available voice styles
- `GET /visual-styles` - Get available visual styles

## Future Enhancements

- Integration with GPT-4o for dialogue generation
- Image generation using GPT Native Image Generator
- 3D model conversion with HunYuan
- Voice generation using ElevenLabs or Bark
- Lip sync with Wav2Lip
- User authentication and project saving
