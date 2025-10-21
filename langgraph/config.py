"""
Environment configuration utility for loading API keys and settings.
"""
import os
from pathlib import Path
from typing import Optional


def load_env_file(env_path: Optional[str] = None) -> None:
    """
    Load environment variables from .env file.
    
    Args:
        env_path: Path to .env file. If None, looks for .env in current directory.
    """
    if env_path is None:
        env_path = Path(__file__).parent.parent / ".env"
    else:
        env_path = Path(env_path)
    
    if not env_path.exists():
        print(f"Warning: .env file not found at {env_path}")
        return
    
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                
                # Only set if not already in environment
                if key not in os.environ:
                    os.environ[key] = value


def get_api_key(service: str) -> str:
    """
    Get API key for a service, with helpful error messages.
    
    Args:
        service: Name of the service (e.g., 'openai', 'anthropic', 'langchain')
    
    Returns:
        API key string
        
    Raises:
        ValueError: If API key is not found
    """
    key_mapping = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'langchain': 'LANGCHAIN_API_KEY',
        'langsmith': 'LANGCHAIN_API_KEY',
        'tavily': 'TAVILY_API_KEY',
        'google': 'GOOGLE_API_KEY',
        'serper': 'SERPER_API_KEY',
        'pinecone': 'PINECONE_API_KEY',
        'weaviate': 'WEAVIATE_API_KEY',
    }
    
    env_var = key_mapping.get(service.lower())
    if not env_var:
        raise ValueError(f"Unknown service: {service}")
    
    api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError(
            f"{env_var} not found in environment variables. "
            f"Please add it to your .env file."
        )
    
    return api_key


# Load environment variables when this module is imported
load_env_file()