from fastapi import FastAPI, File, UploadFile, Body, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any
import uvicorn
import random
import requests
import io
import os
import json
import asyncio
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file if it exists
load_dotenv()

# Get API keys from environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

app = FastAPI(
    title="NPCrafter API",
    description="API for generating context-aware, talking NPCs for games",
    version="0.1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# External API configuration (kept for backward compatibility)
EXTERNAL_API_URL = "http://172.20.10.5:5000"

# OpenAI voice options
OPENAI_VOICES = {
    "alloy": {"gender": "Male (androgynous)", "tone": "Polished, futuristic, balanced"},
    "ash": {"gender": "Male", "tone": "Smoky, grounded, introspective"},
    "ballad": {"gender": "Female", "tone": "Melodic, nostalgic, lyrical"},
    "coral": {"gender": "Female", "tone": "Bright, bubbly, curious"},
    "echo": {"gender": "Female (gender-fluid)", "tone": "Ethereal, ambient, mysterious"},
    "fable": {"gender": "Male", "tone": "Wise, enchanting, storyteller"},
    "nova": {"gender": "Female", "tone": "Bold, radiant, modern"},
    "onyx": {"gender": "Male", "tone": "Deep, luxurious, commanding"},
    "sage": {"gender": "Female", "tone": "Calm, nurturing, insightful"},
    "shimmer": {"gender": "Female", "tone": "Airy, whimsical, radiant"}
}
PERSONA_LIST = [
    {"name": "Barkel", "description": "Low-poly cartoon dog"},
    {"name": "Boneguard", "description": "Armored undead skeleton warrior character"},
    {"name": "Bricktee", "description": "Blocky toy figure with red shirt"},
    {"name": "Drakblade", "description": "Armored dragon character holding a sword"},
    {"name": "Gloam", "description": "Hooded robed figure with glowing eyes"},
    {"name": "Pawvo", "description": "Cartoon dog character waving"},
    {"name": "Plumeca", "description": "Cartoon Aztec child character with feathered headdress"},
    {"name": "Popestep", "description": "Cartoon pope wearing sunglasses and sneakers"},
    {"name": "Steelord", "description": "Cartoon armored knight with sword"}
]

# --- Simplified Models ---

class PersonalityTrait(BaseModel):
    trait: str = Field(..., description="Name of the personality trait")
    intensity: float = Field(1.0, ge=0.0, le=1.0, description="Intensity of the trait (0.0 to 1.0)")
    description: Optional[str] = Field(None, description="Optional free-text description of the trait")

class CreatePersonaRequest(BaseModel):
    npc_name: str = Field(..., description="Name of the NPC")
    game_context: str = Field(..., description="Game world context and setting")
    stage_context: Optional[str] = Field(None, description="Stage or checkpoint description")
    npc_context: str = Field(..., description="NPC's role and position in the game world")
    personality_traits: List[Union[str, PersonalityTrait]] = Field(..., description="List of personality traits or structured traits with intensity")
    dialogue_goal: str = Field(..., description="Dialogue goal (predefined or custom)")
    background: Optional[str] = Field(None, description="Background story of the NPC (optional, will be generated if not provided)")
    language: str = Field("English", description="Preferred language for dialogue")
    generate_voice_sample: bool = Field(True, description="Whether to generate a voice sample")
    sample_text: Optional[str] = Field(None, description="Sample text for voice generation")
    context_file: Optional[UploadFile] = Field(None, description="Optional JSON/state file for in-game context")
    intents: Optional[List[str]] = Field(default_factory=list, description="List of intents for response logic")
    gender: str = Field(..., description="Gender of the NPC (male, female, non-binary, other)", pattern="^(male|female|non-binary|other)$")

class CreatePersonaLangchainRequest(BaseModel):
    """Request model for creating an NPC persona using Langchain and Groq"""
    game_context: str = Field(..., description="Game world context and setting in natural language")
    npc_context: str = Field(..., description="NPC's role, position, and characteristics in natural language")

class SpeakRequest(BaseModel):
    """Request model for generating NPC speech"""
    npc_name: str = Field(..., description="Name of the NPC")
    response: str = Field(..., description="Text to be spoken by the NPC")
    tone: str = Field("neutral", description="Emotional tone of the speech")

class PlayerStats(BaseModel):
    """Single model for all player-related information"""
    health: int = Field(100, description="Current health of the player")
    inventory: List[Dict[str, Any]] = Field(default_factory=list, description="Items the player is carrying")
    completed_quests: List[str] = Field(default_factory=list, description="Quests the player has completed")
    battle_logs: List[Dict[str, Any]] = Field(default_factory=list, description="Record of recent battles")
    achievements: List[str] = Field(default_factory=list, description="Player achievements earned")
    location: str = Field("Unknown", description="Current player location or context")

class NPCTraits(BaseModel):
    """Combined model for all NPC traits and characteristics"""
    name: str = Field(..., description="Name of the NPC")
    personality_traits: List[str] = Field(..., description="List of personality traits")
    dialogue_goal: str = Field(..., description="Purpose or goal of the NPC's dialogue")
    backstory: str = Field("", description="Background story of the NPC")
    visual_style: str = Field("realistic", description="Visual style for the NPC")
    voice_style: Dict[str, str] = Field(default_factory=dict, description="Voice characteristics")
    gender: str = Field(..., description="Gender of the NPC", pattern="^(male|female|non-binary|other)$")

class PersonaCreationResponse(BaseModel):
    npc_name: str = Field(..., description="Name of the NPC")
    personality_traits: List[PersonalityTrait] = Field(..., description="Structured list of personality traits")
    dialogue_goal: str = Field(..., description="Purpose or goal of the NPC's dialogue")
    backstory: str = Field(..., description="Background story of the NPC")
    visual_style: str = Field(..., description="Visual style for the NPC")
    voice_style: Dict[str, str] = Field(..., description="Voice characteristics")
    sample_dialogue: List[str] = Field(..., description="List of sample dialogues from the NPC")
    intents: List[str] = Field(..., description="Recognized intents for response logic")
    memory_id: str = Field(..., description="ID for NPC memory tracking")
    player_context: Dict[str, Any] = Field(default_factory=dict, description="Parsed player context from uploaded file")

class NPCDialogue(BaseModel):
    """Model for NPC dialogue including text, emotion and audio"""
    text: str = Field(..., description="The dialogue text")
    emotion: str = Field("neutral", description="Emotional tone of the dialogue")
    context_references: List[str] = Field(default_factory=list, description="References to player context")
    audio_url: str = Field("", description="URL to the audio version of the dialogue")
    audio_content: Optional[bytes] = Field(None, description="Binary audio content for the dialogue")

class NPCRequest(BaseModel):
    """Request model for generating NPCs"""
    game_world_description: str = Field(..., description="Description of the game world")
    npc_traits: NPCTraits
    preferred_language: str = Field("English", description="Preferred language for dialogue")
    player_stats: Optional[PlayerStats] = None

class NPCResponse(BaseModel):
    """Response model with NPC information"""
    npc_traits: NPCTraits
    dialogue: NPCDialogue
    visual_url: str = Field("", description="URL to the NPC visual")
    model_url: Optional[str] = Field(None, description="URL to the 3D model if available")

# Helper function to select the best persona based on traits
async def select_best_persona(traits: List[str], llm: ChatGroq) -> str:
    """
    Select the best persona from PERSONA_LIST based on NPC personality traits using Groq.
    """
    try:
        prompt_template = ChatPromptTemplate.from_template("""
        You are an expert RPG game designer tasked with selecting the most suitable persona for an NPC based on their personality traits.

        NPC Personality Traits: {traits}

        Available Personas:
        {persona_list}

        Instructions:
        - Analyze the personality traits and match them to the persona descriptions.
        - Choose the persona whose description best aligns with the given traits.
        - Consider the visual and thematic elements implied by the traits (e.g., "mysterious" might suit a hooded figure, "noble" might suit a knight).
        - Return only the name of the selected persona as a string.

        Example Response:
        Drakblade
        """)

        # Format persona list for the prompt
        persona_list_str = "\n".join([f"- {persona['name']}: {persona['description']}" for persona in PERSONA_LIST])

        chain = prompt_template | llm
        result = await chain.ainvoke({
            "traits": ", ".join(traits),
            "persona_list": persona_list_str
        })

        # Extract the persona name from the response
        selected_persona = result.content.strip()
        
        # Validate that the selected persona is in the allowed list
        if selected_persona not in [persona["name"] for persona in PERSONA_LIST]:
            print(f"Invalid persona '{selected_persona}' selected, defaulting to 'Drakblade'")
            selected_persona = "Drakblade"  # Default fallback
        
        return selected_persona

    except Exception as e:
        print(f"Error selecting persona: {str(e)}")
        return "Drakblade"  # Default fallback in case of error


# Helper function to parse uploaded context file
async def parse_context_file(file: UploadFile) -> Dict[str, Any]:
    try:
        content = await file.read()
        if file.filename.endswith('.json'):
            return json.loads(content.decode('utf-8'))
        else:
            return {"raw_content": content.decode('utf-8')}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing context file: {str(e)}")

# Helper function to generate backstory using Groq
async def generate_backstory(npc_name: str, game_context: str, npc_context: str, personality_traits: List[Union[str, PersonalityTrait]], stage_context: Optional[str] = None) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")
    
    llm = ChatGroq(model_name="llama3-8b-8192", api_key=GROQ_API_KEY, temperature=0.7, max_tokens=512)
    
    # Format personality traits for prompt
    traits = []
    for trait in personality_traits:
        if isinstance(trait, PersonalityTrait):
            traits.append(f"{trait.trait} (intensity: {trait.intensity})")
        else:
            traits.append(trait)
    
    prompt_template = ChatPromptTemplate.from_template("""
    You are an expert RPG game designer. Create a detailed backstory for an NPC based on the following:
    
    NPC Name: {npc_name}
    Game Context: {game_context}
    NPC Role: {npc_context}
    Personality Traits: {traits}
    Stage/Checkpoint: {stage_context}
    
    Generate a backstory (3-5 sentences) that:
    1. Fits the game world's setting
    2. Reflects the NPC's role and personality
    3. Incorporates stage/checkpoint context if provided
    4. Is engaging and immersive
    
    Return the backstory as plain text.
    """)
    
    chain = prompt_template | llm
    result = await chain.ainvoke({
        "npc_name": npc_name,
        "game_context": game_context,
        "npc_context": npc_context,
        "traits": ", ".join(traits),
        "stage_context": stage_context or "Not specified"
    })
    
    return result.content.strip()

# Helper function to generate contextual dialogues
async def generate_contextual_dialogues(npc_name: str, game_context: str, npc_context: str, personality_traits: List[Union[str, PersonalityTrait]], stage_context: Optional[str], player_context: Dict[str, Any], intents: List[str], language: str) -> List[str]:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")
    
    llm = ChatGroq(model_name="llama3-8b-8192", api_key=GROQ_API_KEY, temperature=0.75, max_tokens=512)
    
    # Format personality traits
    traits = []
    for trait in personality_traits:
        if isinstance(trait, PersonalityTrait):
            traits.append(f"{trait.trait} (intensity: {trait.intensity})")
        else:
            traits.append(trait)
    
    # Format player context
    player_context_str = json.dumps(player_context, indent=2) if player_context else "No player context provided"
    
    prompt_template = ChatPromptTemplate.from_template("""
    You are an NPC named {npc_name} in a game with the following details:
    
    Game Context: {game_context}
    NPC Role: {npc_context}
    Personality Traits: {traits}
    Stage/Checkpoint: {stage_context}
    Player Context: {player_context}
    Intents: {intents}
    Language: {language}
    
    Generate 3 sample dialogues that:
    1. Reflect the NPC's personality and role
    2. Are appropriate for the game world and stage
    3. Incorporate player context if available
    4. Align with the specified intents
    5. Are in the specified language
    6. Are 1-2 sentences each
    
    Return the dialogues as a JSON array of strings.
    """)
    
    chain = prompt_template | llm | JsonOutputParser()
    result = await chain.ainvoke({
        "npc_name": npc_name,
        "game_context": game_context,
        "npc_context": npc_context,
        "traits": ", ".join(traits),
        "stage_context": stage_context or "Not specified",
        "player_context": player_context_str,
        "intents": ", ".join(intents) or "greeting, assistance, storytelling",
        "language": language
    })
    
    return result

# Helper function to generate intents from context
async def generate_intents(npc_context: str, game_context: str, player_context: Dict[str, Any]) -> List[str]:
    if not GROQ_API_KEY:
        return ["greeting", "assistance"]
    
    llm = ChatGroq(model_name="llama3-8b-8192", api_key=GROQ_API_KEY, temperature=0.7, max_tokens=256)
    
    player_context_str = json.dumps(player_context, indent=2) if player_context else "No player context provided"
    
    prompt_template = ChatPromptTemplate.from_template("""
    Based on the following NPC and game context, identify 3-5 appropriate intents for the NPC's dialogue:
    
    Game Context: {game_context}
    NPC Role: {npc_context}
    Player Context: {player_context}
    
    Intents should be short phrases (e.g., "greeting", "quest-giving", "trading", "storytelling").
    Return the intents as a JSON array of strings.
    """)
    
    chain = prompt_template | llm | JsonOutputParser()
    result = await chain.ainvoke({
        "game_context": game_context,
        "npc_context": npc_context,
        "player_context": player_context_str
    })
    
    return result

# --- Endpoints ---

@app.get("/")
async def root():
    return {"message": "Welcome to NPCrafter API", "version": "0.1.0"}
    
@app.get("/external-api-status")
async def check_external_api():
    """Check if the external API is available"""
    try:
        # Try to ping the external API
        response = requests.get(f"{EXTERNAL_API_URL}/health-check", timeout=5)
        
        if response.status_code == 200:
            return {
                "status": "online",
                "message": "External API is available",
                "url": EXTERNAL_API_URL,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "degraded",
                "message": f"External API returned status code {response.status_code}",
                "url": EXTERNAL_API_URL,
                "timestamp": datetime.now().isoformat()
            }
            
    except requests.RequestException as e:
        return {
            "status": "offline",
            "message": f"Cannot connect to external API: {str(e)}",
            "url": EXTERNAL_API_URL,
            "timestamp": datetime.now().isoformat()
        }

@app.post("/generate-npc", response_model=NPCResponse)
async def generate_npc(request: NPCRequest):
    """Generate an NPC based on the provided parameters"""
    
    # Dummy NPC generation
    emotions = ["happy", "sad", "angry", "neutral", "surprised"]
    
    # Reference player stats if available
    context_refs = []
    if request.player_stats:
        if request.player_stats.health < 50:
            context_refs.append("low health")
        if request.player_stats.achievements:
            context_refs.append(f"achievement: {random.choice(request.player_stats.achievements)}")
        if request.player_stats.completed_quests:
            context_refs.append(f"quest: {random.choice(request.player_stats.completed_quests)}")
        if request.player_stats.location:
            context_refs.append(f"location: {request.player_stats.location}")
    
    # Create dialogue text
    dialogue_text = f"Greetings, traveler! I am {request.npc_traits.name}. I notice you've been through {request.game_world_description}. That's quite impressive."
    emotion = random.choice(emotions)
    
    # Enhance NPC traits with backstory if not provided
    if not request.npc_traits.backstory:
        request.npc_traits.backstory = f"A veteran of many adventures, now {request.npc_traits.dialogue_goal} all travelers passing through."
    
    # Add voice style if not provided
    if not request.npc_traits.voice_style:
        request.npc_traits.voice_style = {"tone": "deep", "accent": "slight fantasy", "pace": "measured"}
    
    # Determine tone for voice generation based on emotion
    tone_mapping = {
        "happy": "cheerful",
        "sad": "somber",
        "angry": "stern",
        "neutral": "neutral",
        "surprised": "surprised"
    }
    tone = tone_mapping.get(emotion, "neutral")
    
    # Audio URL - try to generate from external API
    audio_url = "https://dummy-audio-url.com/npc-voice.mp3"  # Default fallback
    try:
        # Attempt to generate real audio via external API
        external_api_url = f"{EXTERNAL_API_URL}/npc/speak"
        payload = {
            "npc_name": request.npc_traits.name,
            "response": dialogue_text,
            "tone": tone
        }
        
        # For now, don't actually make the request in this endpoint to avoid blocking
        # Just note that the audio would be generated in a real implementation
        # We would use the helper method: await get_audio_from_external_api(external_api_url, payload)
        
        # In a real implementation with file storage:
        # 1. Store the audio file
        # 2. Generate a URL for the stored file
        # 3. Set audio_url to the generated URL
    except Exception as e:
        # Log the error but continue with dummy URL
        print(f"Error generating audio: {str(e)}")
    
    # Create dialogue object
    dialogue = NPCDialogue(
        text=dialogue_text,
        emotion=emotion,
        context_references=context_refs,
        audio_url=audio_url
    )
    
    return NPCResponse(
        npc_traits=request.npc_traits,
        dialogue=dialogue,
        visual_url="https://dummy-image-url.com/npc-image.png",
        model_url="https://dummy-model-url.com/npc-model.glb" if random.choice([True, False]) else None
    )

@app.post("/interact-with-npc")
async def interact_with_npc(
    npc_id: str = Query(..., description="ID of the generated NPC"),
    message: str = Body(..., embed=True, description="Player's message to the NPC"),
    player_stats: Optional[PlayerStats] = Body(None, embed=True)
):
    """Interact with a previously generated NPC"""
    
    emotions = ["happy", "contemplative", "curious", "amused"]
    emotion = random.choice(emotions)
    
    # Dummy response
    context_refs = []
    if player_stats:
        if player_stats.health < 30:
            context_refs.append("critical health")
        elif player_stats.health < 70:
            context_refs.append("wounded")
        if player_stats.location:
            context_refs.append(f"location: {player_stats.location}")
    
    # Get NPC details - in a real implementation, this would fetch from a database
    # For this dummy implementation, we'll use the npc_id as the NPC name
    npc_name = f"NPC-{npc_id}"
    
    # Create response text
    response_text = f"I hear your words, adventurer. {message[:20]}... yes, that's interesting."
    
    # Determine tone for voice generation based on emotion
    tone_mapping = {
        "happy": "cheerful",
        "contemplative": "thoughtful",
        "curious": "intrigued",
        "amused": "playful"
    }
    tone = tone_mapping.get(emotion, "neutral")
    
    # Audio URL - try to generate from external API
    audio_url = "https://dummy-audio-url.com/response.mp3"  # Default fallback
    try:
        # Attempt to generate real audio via external API
        external_api_url = f"{EXTERNAL_API_URL}/npc/speak"
        payload = {
            "npc_name": npc_name,
            "response": response_text,
            "tone": tone
        }
        
        # For now, don't actually make the request in this endpoint to avoid blocking
        # Just note that the audio would be generated in a real implementation
        # We would use the helper method: await get_audio_from_external_api(external_api_url, payload)
        
        # In a real implementation with file storage:
        # 1. Store the audio file
        # 2. Generate a URL for the stored file
        # 3. Set audio_url to the generated URL
    except Exception as e:
        # Log the error but continue with dummy URL
        print(f"Error generating audio: {str(e)}")
    
    dialogue = NPCDialogue(
        text=response_text,
        emotion=emotion,
        context_references=context_refs,
        audio_url=audio_url
    )
    
    return {
        "npc_id": npc_id,
        "dialogue": dialogue.dict(),
        "animation": "talking",
        "audio_generation_status": "completed" # In a real implementation, this would indicate if audio was generated
    }

@app.post("/upload-game-data")
async def upload_game_data(
    game_file: UploadFile = File(...),
    file_type: str = Query(..., description="Type of file being uploaded (e.g., 'state', 'logs')")
):
    """Upload game data files for NPC context"""
    
    # Dummy processing
    file_content = await game_file.read()
    file_size = len(file_content)
    
    return {
        "filename": game_file.filename,
        "type": file_type,
        "size": file_size,
        "processed": True,
        "upload_time": datetime.now().isoformat(),
        "status": "success"
    }

@app.get("/export-npc/{npc_id}")
async def export_npc(
    npc_id: str,
    format: str = Query("all", description="Export format: 'all', 'visual', 'audio', 'model', 'dialogue', 'traits'")
):
    """Export generated NPC assets"""
    
    formats = {
        "visual": ["png", "jpg"],
        "audio": ["mp3", "wav"],
        "model": ["glb"],
        "dialogue": ["json", "txt"],
        "traits": ["json"],
        "all": ["png", "mp3", "glb", "json"]
    }
    
    if format not in formats:
        raise HTTPException(status_code=400, detail="Invalid export format")
    
    # Dummy NPC data for export
    npc_data = {
        "npc_traits": {
            "name": "Example NPC",
            "personality_traits": ["wise", "helpful", "mysterious"],
            "dialogue_goal": "guide adventurers",
            "backstory": "A seasoned explorer who has seen many worlds.",
            "visual_style": "realistic",
            "voice_style": {"tone": "deep", "pace": "measured"}
        },
        "dialogue_samples": [
            {
                "text": "Greetings, traveler! What brings you to these parts?",
                "emotion": "friendly",
                "audio_url": "https://dummy-url.com/npc-greeting.mp3"
            },
            {
                "text": "Be careful in the forest ahead. Strange things lurk there.",
                "emotion": "concerned",
                "audio_url": "https://dummy-url.com/npc-warning.mp3"
            }
        ]
    }
    
    return {
        "npc_id": npc_id,
        "export_formats": formats[format],
        "download_urls": {
            "png": "https://dummy-url.com/npc-visual.png",
            "mp3": "https://dummy-url.com/npc-voice-pack.mp3",
            "glb": "https://dummy-url.com/npc-model.glb",
            "json": "https://dummy-url.com/npc-data.json"
        },
        "npc_data": npc_data if format in ["traits", "all"] else None,
        "export_time": datetime.now().isoformat()
    }

@app.get("/voice-styles")
async def get_voice_styles():
    """Get available voice styles for NPC generation"""
    
    return {
        "styles": [
            {"id": "wise_elder", "name": "Wise Elder", "description": "Deep, slow, thoughtful voice"},
            {"id": "merchant", "name": "Eager Merchant", "description": "Fast-talking, enthusiastic"},
            {"id": "warrior", "name": "Battle-hardened Warrior", "description": "Gruff, commanding"},
            {"id": "mystic", "name": "Mystic Oracle", "description": "Ethereal, mysterious"},
            {"id": "villager", "name": "Simple Villager", "description": "Friendly, simple dialect"}
        ]
    }

@app.get("/visual-styles")
async def get_visual_styles():
    """Get available visual styles for NPC generation"""
    
    return {
        "styles": [
            {"id": "anime", "name": "Anime", "description": "Japanese animation style"},
            {"id": "pixel", "name": "Pixel Art", "description": "Retro pixel art style"},
            {"id": "realistic", "name": "Realistic", "description": "Photorealistic rendering"},
            {"id": "comic", "name": "Comic/Cartoon", "description": "Western comic book style"}
        ]
    }

@app.post("/random-npc", response_model=NPCResponse)
async def create_random_npc(
    game_world: str = Body("Fantasy Kingdom", embed=True),
    player_stats: Optional[PlayerStats] = Body(None, embed=True)
):
    """Create a random NPC with minimal input"""
    
    # Sample NPC names and traits
    npc_names = ["Eldrin", "Zara", "Grimlock", "Seraphina", "Thorne", "Aria", "Kael", "Lyra"]
    personality_traits = [
        ["wise", "patient", "knowledgeable"],
        ["cheerful", "optimistic", "helpful"],
        ["gruff", "battle-hardened", "direct"],
        ["mysterious", "cryptic", "ancient"],
        ["friendly", "talkative", "curious"]
    ]
    dialogue_goals = ["guide adventurers", "trade rare goods", "share wisdom", "test worthy heroes", "collect stories"]
    visual_styles = ["anime", "pixel-art", "realistic", "comic"]
    
    # Generate random NPC traits
    npc_traits = NPCTraits(
        name=random.choice(npc_names),
        personality_traits=random.choice(personality_traits),
        dialogue_goal=random.choice(dialogue_goals),
        backstory=f"A long-time resident of {game_world}, with many tales to tell.",
        visual_style=random.choice(visual_styles),
        voice_style={"tone": random.choice(["deep", "melodic", "raspy", "soft"]), "pace": random.choice(["quick", "measured", "slow"])}
    )
    
    # Generate dialogue
    emotions = ["neutral", "happy", "curious", "wise"]
    context_refs = []
    
    # Add context based on player stats if provided
    if player_stats:
        if player_stats.health < 50:
            context_refs.append("low health")
        if player_stats.achievements:
            context_refs.append(f"achievement: {random.choice(player_stats.achievements)}")
        if player_stats.location:
            context_refs.append(f"location: {player_stats.location}")
    
    dialogue = NPCDialogue(
        text=f"Welcome to {game_world}, traveler. I am {npc_traits.name}, and I {npc_traits.dialogue_goal}.",
        emotion=random.choice(emotions),
        context_references=context_refs,
        audio_url="https://dummy-audio-url.com/npc-voice.mp3"
    )
    
    return NPCResponse(
        npc_traits=npc_traits,
        dialogue=dialogue,
        visual_url="https://dummy-image-url.com/npc-image.png",
        model_url="https://dummy-model-url.com/npc-model.glb" if random.choice([True, False]) else None
    )
@app.post("/preview-dialogue")
async def preview_dialogue(
    npc_traits: NPCTraits = Body(..., description="NPC traits to base dialogue on"),
    message: str = Body(None, embed=True, description="Optional player message to respond to"),
    player_stats: Optional[PlayerStats] = Body(None, embed=True, description="Optional player stats for context"),
    history: Optional[List[str]] = Body(None, embed=True, description="Optional dialogue history for context")
):
    """Generate a preview of NPC dialogue based on NPC traits and optional player input using Groq API"""
    
    try:
        if not GROQ_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="GROQ_API_KEY environment variable not set"
            )
        
        # Initialize the Groq LLM through Langchain
        llm = ChatGroq(
            model_name="llama3-8b-8192",
            api_key=GROQ_API_KEY,
            temperature=0.75,
            max_tokens=512
        )
        
        # Initialize context references list
        context_refs = []
        
        # Build player context string for the prompt
        player_context_parts = []
        
        if player_stats:
            # Health context
            if player_stats.health < 30:
                context_refs.append("critical health")
                player_context_parts.append("The player appears gravely wounded and in critical condition.")
            elif player_stats.health < 70:
                context_refs.append("wounded")
                player_context_parts.append("The player has some visible injuries.")
            else:
                player_context_parts.append("The player appears to be in good health.")
            
            # Location context
            if player_stats.location:
                context_refs.append(f"location: {player_stats.location}")
                player_context_parts.append(f"You are currently in {player_stats.location}.")
            
            # Quest context
            if player_stats.completed_quests:
                context_refs.append(f"completed quests: {len(player_stats.completed_quests)}")
                quest_list = ', '.join(player_stats.completed_quests[:3])
                player_context_parts.append(f"The player has completed these quests: {quest_list}")
            
            # Inventory context
            if player_stats.inventory:
                context_refs.append(f"carrying {len(player_stats.inventory)} items")
                items = [item.get('name', 'unknown item') for item in player_stats.inventory[:3]]
                player_context_parts.append(f"The player is carrying: {', '.join(items)}")
            
            # Achievement context
            if player_stats.achievements:
                context_refs.append(f"achievements: {len(player_stats.achievements)}")
                achievements = ', '.join(player_stats.achievements[:2])
                player_context_parts.append(f"The player has earned achievements: {achievements}")
        
        # Create the prompt template for dialogue generation
        prompt_template = ChatPromptTemplate.from_template("""
        You are roleplaying as an NPC in a game with the following characteristics:
        
        Character Name: {name}
        Personality Traits: {traits}
        Dialogue Goal: {goal}
        Backstory: {backstory}
        Voice Style: {voice_style}
        History: {history}
        
        Context Information:
        {player_context}
        
        {interaction_prompt}
        
        Generate a response that:
        1. Stays true to your character's personality and goals
        2. References the player's context when appropriate
        3. Is 2-4 sentences long
        4. Sounds natural and engaging
        5. Advances the conversation or provides value to the player
        
        Also determine the most appropriate emotion for this response from: neutral, happy, concerned, curious, amused, thoughtful, stern, surprised, friendly, mysterious
        
        Format your response EXACTLY as follows, with no additional text, explanations, or notes:
        EMOTION: [selected emotion]
        DIALOGUE: [your response as the character]
        """)
        
        # Prepare voice style string
        voice_style = "normal speaking voice"
        if npc_traits.voice_style:
            voice_style_parts = []
            for key, value in npc_traits.voice_style.items():
                voice_style_parts.append(f"{key}: {value}")
            if voice_style_parts:
                voice_style = ", ".join(voice_style_parts)
        
        # Determine interaction prompt based on whether player sent a message
        if message:
            interaction_prompt = f'The player says to you: "{message}"\n\nRespond to the player\'s message as your character would.'
        else:
            interaction_prompt = "Generate an opening dialogue that your character would say when first meeting the player."
        
        # Combine player context
        player_context = "\n".join(player_context_parts) if player_context_parts else "No specific player context available."
        
        # Generate the dialogue using Groq
        chain_result = llm.invoke(prompt_template.format(
            name=npc_traits.name,
            traits=", ".join(npc_traits.personality_traits),
            goal=npc_traits.dialogue_goal,
            backstory=npc_traits.backstory if npc_traits.backstory else "A character with a mysterious past.",
            voice_style=voice_style,
            player_context=player_context,
            interaction_prompt=interaction_prompt
        ))
        
        # Parse the LLM response
        response_text = chain_result.content.strip()

        # Default values
        emotion = "neutral"
        dialogue_text = response_text

        valid_emotions = ["neutral", "happy", "concerned", "curious", "amused", "thoughtful", "stern", "surprised", "friendly", "mysterious"]
        # Try to parse the structured response
        if "EMOTION:" in response_text and "DIALOGUE:" in response_text:
            try:
                # Split into emotion and dialogue parts
                parts = response_text.split("DIALOGUE:", 1)
                emotion_part = parts[0].split("EMOTION:", 1)[1].strip().lower()
                dialogue_text = parts[1].strip()
                
                # Remove any trailing notes or extra content (e.g., "(Note: ...)") from dialogue_text
                if "\n\n" in dialogue_text:
                    dialogue_text = dialogue_text.split("\n\n")[0].strip()
                
                # Validate emotion
                if emotion_part in valid_emotions:
                    emotion = emotion_part
                else:
                    print(f"Invalid emotion '{emotion_part}' detected, defaulting to 'neutral'")
            except Exception as e:
                print(f"Error parsing response: {str(e)}")
                # Fall back to treating the entire response as dialogue, but try to clean it up
                dialogue_text = response_text.split("\n\n")[0].strip() if "\n\n" in response_text else response_text
        else:
            print("Response does not contain EMOTION and DIALOGUE markers, treating as dialogue")
            # Try to clean up by removing any known prefix like "THOUGHTFUL:"
            for valid_emotion in valid_emotions:
                prefix = f"{valid_emotion.upper()}:"
                if response_text.startswith(prefix):
                    dialogue_text = response_text[len(prefix):].strip()
                    emotion = valid_emotion.lower()
                    break
            # Remove any trailing notes
            dialogue_text = dialogue_text.split("\n\n")[0].strip() if "\n\n" in dialogue_text else dialogue_text
        
        # Map emotion to tone for voice generation
        emotion_to_tone_mapping = {
            "happy": "cheerful",
            "concerned": "somber",
            "curious": "intrigued", 
            "amused": "playful",
            "thoughtful": "thoughtful",
            "stern": "stern",
            "surprised": "intrigued",
            "friendly": "cheerful",
            "mysterious": "thoughtful",
            "neutral": "neutral"
        }
        tone = emotion_to_tone_mapping.get(emotion, "neutral")
        
        # Generate audio using OpenAI TTS
        audio_content = None
        audio_url = ""
        audio_generation_success = False
        
        try:
            if OPENAI_API_KEY:
                # Select the best voice for the emotion and NPC traits
                voice = select_openai_voice(emotion, npc_traits)
                
                # Generate speech using OpenAI
                audio_content = generate_speech_with_openai(
                    text=dialogue_text,
                    voice=voice,
                    emotion=emotion
                )
                
                # Create filename for file serving
                timestamp = int(datetime.now().timestamp())
                filename = f"{npc_traits.name.replace(' ', '_')}_{timestamp}.mp3"
                
                # Save the audio file
                os.makedirs("audio_files", exist_ok=True)
                file_path = os.path.join("audio_files", filename)
                with open(file_path, "wb") as f:
                    f.write(audio_content)
                
                # Set audio URL
                audio_url = f"/audio/{filename}"
                audio_generation_success = True
                print(f"Generated audio with OpenAI using voice '{voice}' and saved to {file_path}")
            else:
                print("OpenAI API key not set, skipping audio generation")
                
        except Exception as e:
            print(f"Error generating audio with OpenAI: {str(e)}")
            audio_generation_success = False
        
        # Create the dialogue response
        dialogue = NPCDialogue(
            text=dialogue_text,
            emotion=emotion,
            context_references=context_refs,
            audio_url=audio_url,
            audio_content=audio_content
        )

        history.append(dialogue_text.text)
        
        # Prepare the response data
        dialogue_dict = dialogue.dict()
        if 'audio_content' in dialogue_dict:
            del dialogue_dict['audio_content']  # Remove binary data from JSON response
        
        response_data = {
            "dialogue": dialogue_dict,
            "llm_model": "llama3-8b-8192",
            "context_used": len(context_refs) > 0,
            "audio_generation_status": "success" if audio_generation_success else "failed",
            "message": "Dialogue and audio generated successfully" if audio_generation_success else "Dialogue generated successfully, but audio generation failed",
            "history": history
        }
        
        # If audio was generated successfully, include it in the response
        if audio_generation_success and audio_content:
            import base64
            response_data["audio_base64"] = base64.b64encode(audio_content).decode('utf-8')
            response_data["audio_format"] = "mp3"
        
        return response_data
        
    except Exception as e:
        # Handle any errors that occur during processing
        print(f"Error in preview_dialogue: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating dialogue: {str(e)}"
        )

@app.post("/save-npc")
async def save_npc(
    npc_response: NPCResponse,
    npc_id: Optional[str] = Body(None, embed=True)
):
    """Save or update an NPC in the database"""
    
    # Generate a random ID if not provided
    if not npc_id:
        npc_id = f"npc-{random.randint(1000, 9999)}"
    
    # In a real application, this would save to a database
    return {
        "message": "NPC saved successfully",
        "npc_id": npc_id,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/npc/{npc_id}")
async def get_npc(npc_id: str):
    """Retrieve an NPC by ID"""
    
    # In a real application, this would fetch from a database
    # For now, return dummy data
    
    emotions = ["neutral", "happy", "concerned", "curious"]
    visual_styles = ["anime", "pixel-art", "realistic", "comic"]
    
    npc_traits = NPCTraits(
        name=f"NPC-{npc_id}",
        personality_traits=["mysterious", "helpful", "wise"],
        dialogue_goal="guide adventurers",
        backstory="A character with a rich history in these lands.",
        visual_style=random.choice(visual_styles),
        voice_style={"tone": "deep", "accent": "fantasy", "pace": "measured"}
    )
    
    dialogue = NPCDialogue(
        text="Ah, I see you've returned. What adventures have you had since we last spoke?",
        emotion=random.choice(emotions),
        context_references=[],
        audio_url="https://dummy-audio-url.com/npc-greeting.mp3"
    )
    
    return NPCResponse(
        npc_traits=npc_traits,
        dialogue=dialogue,
        visual_url="https://dummy-image-url.com/npc-image.png",
        model_url="https://dummy-model-url.com/npc-model.glb" if random.choice([True, False]) else None
    )

@app.post("/npc/create-persona", response_model=PersonaCreationResponse)
async def create_persona(
    request: CreatePersonaRequest = Body(...),
    voice: str = Query(None, description="Optional override for the OpenAI voice to use"),
    context_file: Optional[UploadFile] = File(None, description="Optional JSON/state file for in-game context")
):
    """Create an NPC persona with voice sample and enhanced features using OpenAI TTS and Groq LLM"""
    try:
        if not GROQ_API_KEY:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")
        if request.generate_voice_sample and not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set for voice generation")
        
        # Parse personality traits into structured format
        personality_traits = []
        for trait in request.personality_traits:
            if isinstance(trait, str):
                personality_traits.append(PersonalityTrait(trait=trait, intensity=1.0))
            else:
                personality_traits.append(trait)
        
        # Parse uploaded context file
        player_context = {}
        if context_file:
            player_context = await parse_context_file(context_file)
        elif request.context_file:
            player_context = await parse_context_file(request.context_file)
        
        # Generate backstory if not provided
        backstory = request.background
        if not backstory:
            backstory = await generate_backstory(
                npc_name=request.npc_name,
                game_context=request.game_context,
                npc_context=request.npc_context,
                personality_traits=personality_traits,
                stage_context=request.stage_context
            )
        
        # Generate or use provided intents
        intents = request.intents
        if not intents:
            intents = await generate_intents(
                npc_context=request.npc_context,
                game_context=request.game_context,
                player_context=player_context
            )
        
        # Generate contextual dialogues
        sample_dialogues = await generate_contextual_dialogues(
            npc_name=request.npc_name,
            game_context=request.game_context,
            npc_context=request.npc_context,
            personality_traits=personality_traits,
            stage_context=request.stage_context,
            player_context=player_context,
            intents=intents,
            language=request.language
        )
        
        # Determine emotion based on personality traits
        emotion = "neutral"
        trait_names = [t.trait.lower() for t in personality_traits]
        if any(t in ["cheerful", "happy", "optimistic"] for t in trait_names):
            emotion = "happy"
        elif any(t in ["wise", "thoughtful", "contemplative"] for t in trait_names):
            emotion = "thoughtful"
        elif any(t in ["mysterious", "enigmatic", "cryptic"] for t in trait_names):
            emotion = "mysterious"
        elif any(t in ["stern", "serious", "commanding"] for t in trait_names):
            emotion = "stern"
        
        # Generate voice style
        voice_style = {
            "tone": emotion,
            "accent": "neutral" if request.language.lower() == "english" else request.language.lower(),
            "pace": "measured",
            "gender": request.gender
        }
        
        # Select visual style
        visual_styles = ["realistic", "anime", "pixel-art", "comic"]
        visual_style = random.choice(visual_styles)
        
        # Generate or use sample text for voice
        sample_text = request.sample_text
        if not sample_text and sample_dialogues:
            sample_text = sample_dialogues[0]
        
        # Generate voice sample if requested
        audio_url = ""
        if request.generate_voice_sample and sample_text:
            selected_voice = voice if voice else select_openai_voice(emotion, NPCTraits(
                name=request.npc_name,
                personality_traits=[t.trait for t in personality_traits],
                dialogue_goal=request.dialogue_goal,
                backstory=backstory,
                voice_style=voice_style,
                gender=request.gender
            ))
            
            audio_content = generate_speech_with_openai(
                text=sample_text,
                voice=selected_voice,
                emotion=emotion
            )
            
            timestamp = int(datetime.now().timestamp())
            filename = f"{request.npc_name.replace(' ', '_')}_{timestamp}.mp3"
            os.makedirs("audio_files", exist_ok=True)
            file_path = os.path.join("audio_files", filename)
            with open(file_path, "wb") as f:
                f.write(audio_content)
            
            audio_url = f"/audio/{filename}"
        
        # Generate memory ID and store NPC data
        memory_id = f"npc-{random.randint(1000, 9999)}"
        NPC_MEMORY[memory_id] = {
            "npc_name": request.npc_name,
            "game_context": request.game_context,
            "stage_context": request.stage_context,
            "npc_context": request.npc_context,
            "personality_traits": personality_traits,
            "dialogue_goal": request.dialogue_goal,
            "backstory": backstory,
            "intents": intents,
            "player_context": player_context,
            "interactions": [],
            "gender": request.gender
        }
        
        # Create response
        response = PersonaCreationResponse(
            npc_name=request.npc_name,
            personality_traits=personality_traits,
            dialogue_goal=request.dialogue_goal,
            backstory=backstory,
            visual_style=visual_style,
            voice_style=voice_style,
            sample_dialogue=sample_dialogues,
            intents=intents,
            memory_id=memory_id,
            player_context=player_context
        )
        
        # If audio was generated, return it directly
        if request.generate_voice_sample and audio_url:
            with open(file_path, "rb") as f:
                audio_content = f.read()
            return Response(
                content=audio_content,
                media_type="audio/mp3",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"',
                    "X-NPC-Metadata": json.dumps(response.dict())
                }
            )
        
        return response
    
    except Exception as e:
        print(f"Error in create_persona: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating persona: {str(e)}")

# Updated /npc/create-persona-langchain endpoint
@app.post("/npc/create-persona-langchain", response_model=PersonaCreationResponse)
async def create_persona_langchain(request: CreatePersonaLangchainRequest):
    """Create an NPC persona using Langchain and Groq from natural language descriptions, and select the best persona ID."""
    try:
        if not GROQ_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="GROQ_API_KEY environment variable not set"
            )
        
        # Initialize the Groq LLM through Langchain
        llm = ChatGroq(
            model_name="llama3-8b-8192",
            api_key=GROQ_API_KEY,
            temperature=0.7,
            max_tokens=1024
        )
        
        # Create a parser to get structured output
        parser = JsonOutputParser(pydantic_object=PersonaCreationResponse)
        
        # Define the prompt template for persona creation
        gender_instruction = f"Gender: {request.gender}\n" if hasattr(request, 'gender') and request.gender else "Gender: Choose an appropriate gender based on the context (male, female, non-binary, or other).\n"
        prompt_template = ChatPromptTemplate.from_template("""
        You are an expert RPG game designer and storyteller. 
        Your task is to create a detailed NPC persona based on the provided game context and NPC description.
        
        Game Context:
        {game_context}
        
        NPC Description:
        {npc_context}
        
        {gender_instruction}
        
        Create a comprehensive NPC persona with the following details:
        1. A fitting name for the NPC
        2. 3-5 personality traits that define the character
        3. The dialogue goal or purpose of this NPC in the game
        4. A rich backstory consistent with the game world
        5. Visual style recommendation (realistic, anime, pixel art, or comic)
        6. Voice style characteristics including tone, accent, and pace
        7. A sample dialogue line this NPC might say
        8. The NPC's gender (male, female, non-binary, or other)
        
        Provide your response as a JSON object with the following structure:
        {format_instructions}
        """)
        
        # Create the chain
        chain = (
            prompt_template
            | llm
            | parser
        )
        
        # Execute the chain with the user's input
        result = await chain.ainvoke({
            "game_context": request.game_context,
            "npc_context": request.npc_context,
            "gender_instruction": gender_instruction,
            "format_instructions": parser.get_format_instructions()
        })
        
        # Select the best persona based on the generated personality traits
        selected_persona = await select_best_persona(result["personality_traits"], llm)
        
        # Add the selected persona to the response
        result["selected_persona"] = selected_persona
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error creating persona with Langchain: {str(e)}"
        )

@app.post("/npc/speak")
async def npc_speak(
    request: SpeakRequest,
    voice: str = Query(None, description="Optional override for the OpenAI voice to use")
):
    """Generate NPC speech using OpenAI's TTS API"""
    try:
        if not OPENAI_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY environment variable not set"
            )
        
        # Map tone to emotion
        tone_to_emotion = {
            "cheerful": "happy",
            "somber": "concerned",
            "intrigued": "curious",
            "playful": "amused",
            "thoughtful": "thoughtful",
            "stern": "stern",
            "neutral": "neutral"
        }
        
        emotion = tone_to_emotion.get(request.tone.lower(), "neutral")
        
        # Select voice if not provided
        selected_voice = voice if voice else select_openai_voice(emotion)
        
        # Generate audio using OpenAI
        audio_content = generate_speech_with_openai(
            text=request.response,
            voice=selected_voice,
            emotion=emotion
        )
        
        # Create a timestamp-based filename
        timestamp = int(datetime.now().timestamp())
        filename = f"{request.npc_name.replace(' ', '_')}_{timestamp}.mp3"
        
        # Create directory for audio files if it doesn't exist
        os.makedirs("audio_files", exist_ok=True)
        
        # Save the audio file
        file_path = os.path.join("audio_files", filename)
        with open(file_path, "wb") as f:
            f.write(audio_content)
        
        # Return the audio file directly in response body
        return Response(
            content=audio_content,
            media_type="audio/mp3",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    except Exception as e:
        # Handle errors
        print(f"Error in npc_speak: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating speech: {str(e)}"
        )

def select_openai_voice(emotion: str, npc_traits: NPCTraits = None):
    """
    Select the most appropriate OpenAI voice based on emotion and NPC traits
    """
    # Map emotions to voice recommendations
    emotion_to_voice = {
        "neutral": ["alloy", "sage"],
        "happy": ["coral", "nova", "shimmer"],
        "concerned": ["ash", "sage"],
        "curious": ["coral", "echo"],
        "amused": ["coral", "shimmer"],
        "thoughtful": ["sage", "ash", "fable"],
        "stern": ["onyx", "alloy"],
        "surprised": ["shimmer", "nova"],
        "friendly": ["coral", "ballad"],
        "mysterious": ["echo", "fable"],
        "wise": ["fable", "sage"],
        "commanding": ["onyx", "alloy"],
        "somber": ["ash", "ballad"],
        "playful": ["coral", "shimmer"]
    }
    
    # Get voice options for the emotion
    voice_options = emotion_to_voice.get(emotion.lower(), ["alloy"])
    
    # If NPC traits are provided, use them to refine the selection
    if npc_traits:
        filtered_voices = []
        gender_pref = npc_traits.gender.lower() if npc_traits.gender else None
        
        if gender_pref:
            for voice in voice_options:
                voice_gender = OPENAI_VOICES[voice]["gender"].lower()
                # Match gender, accounting for partial matches (e.g., "male (androgynous)" matches "male")
                if gender_pref in voice_gender or (gender_pref == "non-binary" and "gender-fluid" in voice_gender):
                    filtered_voices.append(voice)
            
            if filtered_voices:
                voice_options = filtered_voices
        # Consider voice_style if no specific gender match
        elif npc_traits.voice_style and "gender" in npc_traits.voice_style:
            gender_pref = npc_traits.voice_style["gender"].lower()
            for voice in voice_options:
                voice_gender = OPENAI_VOICES[voice]["gender"].lower()
                if gender_pref in voice_gender or (gender_pref == "non-binary" and "gender-fluid" in voice_gender):
                    filtered_voices.append(voice)
            
            if filtered_voices:
                voice_options = filtered_voices
    
    # Return the first voice in the options list
    return voice_options[0]

# Helper method to generate audio using OpenAI TTS API
def generate_speech_with_openai(text: str, voice: str, emotion: str = "neutral"):
    """
    Generate speech audio using OpenAI's TTS API
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Map emotion to instruction
        emotion_to_instruction = {
            "neutral": "Speak in a natural, balanced tone.",
            "happy": "Speak in a cheerful, positive tone.",
            "concerned": "Speak in a concerned, caring tone.",
            "curious": "Speak in an inquisitive, interested tone.",
            "amused": "Speak in an amused, slightly playful tone.",
            "thoughtful": "Speak in a thoughtful, contemplative tone.",
            "stern": "Speak in a serious, authoritative tone.",
            "surprised": "Speak in a surprised, slightly elevated tone.",
            "friendly": "Speak in a warm, welcoming tone.",
            "mysterious": "Speak in a mysterious, enigmatic tone.",
            "wise": "Speak in a wise, knowledgeable tone.",
            "commanding": "Speak in a commanding, authoritative tone.",
            "somber": "Speak in a somber, serious tone.",
            "playful": "Speak in a playful, lighthearted tone."
        }
        
        instructions = emotion_to_instruction.get(emotion.lower(), "Speak in a natural tone.")
        
        # Create speech audio
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            response_format="mp3",
            speed=1.0,
            instructions=instructions
        )
        
        # Get the audio content
        audio_content = response.content
        
        return audio_content
        
    except Exception as e:
        print(f"OpenAI speech generation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating speech: {str(e)}"
        )

# Route to serve generated audio files
@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """Serve saved audio files"""
    try:
        file_path = os.path.join("audio_files", filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Audio file '{filename}' not found")
        
        # Determine media type based on file extension
        media_type = "audio/mp3" if filename.endswith(".mp3") else "audio/wav"
        
        # Read the audio file content
        with open(file_path, "rb") as f:
            audio_content = f.read()
            
        # Return the audio file content directly in the response body
        return Response(
            content=audio_content,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error serving audio file: {str(e)}")
        
@app.get("/npc/voice-options")
async def get_voice_options():
    """Get available voice options for NPCs using OpenAI voices"""
    
    # Format the OpenAI voices for the response
    voices = []
    for voice_id, details in OPENAI_VOICES.items():
        voices.append({
            "id": voice_id,
            "name": voice_id.capitalize(),
            "gender": details["gender"],
            "tone": details["tone"],
            "description": f"{details['gender']} voice with {details['tone']} characteristics"
        })
    
    # Define emotion to tone mapping for the response
    emotions = [
        {"id": "neutral", "name": "Neutral", "description": "Standard speaking voice"},
        {"id": "happy", "name": "Happy", "description": "Cheerful and positive"},
        {"id": "concerned", "name": "Concerned", "description": "Caring and worried"},
        {"id": "curious", "name": "Curious", "description": "Inquisitive and interested"},
        {"id": "amused", "name": "Amused", "description": "Slightly playful and entertained"},
        {"id": "thoughtful", "name": "Thoughtful", "description": "Contemplative and wise"},
        {"id": "stern", "name": "Stern", "description": "Serious and authoritative"},
        {"id": "surprised", "name": "Surprised", "description": "Astonished and amazed"},
        {"id": "friendly", "name": "Friendly", "description": "Warm and welcoming"},
        {"id": "mysterious", "name": "Mysterious", "description": "Enigmatic and intriguing"},
        {"id": "wise", "name": "Wise", "description": "Knowledgeable and sagely"},
        {"id": "commanding", "name": "Commanding", "description": "Authoritative and powerful"},
        {"id": "somber", "name": "Somber", "description": "Sad and serious"},
        {"id": "playful", "name": "Playful", "description": "Fun and lighthearted"}
    ]
    
    # Return all available options
    return {
        "voices": voices,
        "emotions": emotions,
        "models": [
            {"id": "tts-1", "name": "TTS-1", "description": "Standard text-to-speech model"},
            {"id": "tts-1-hd", "name": "TTS-1-HD", "description": "High-definition text-to-speech model"}
        ],
        "default_voice": "alloy",
        "default_emotion": "neutral",
        "source": "openai"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
