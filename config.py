"""
config.py — Configuration globale du pipeline Riot Draft.

Toutes les constantes de routing, queue IDs et chemins sont centralisées ici.
"""
from pathlib import Path

# ── Répertoires ──────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).parent
DATA_DIR: Path = BASE_DIR / "data"
DB_PATH: Path = DATA_DIR / "draft.db"

# ── Riot API Routing ─────────────────────────────────────────────────────────
# Platform routing (Summoner-V4, League-V4) — spécifique à la région physique
PLATFORM: str = "euw1"
PLATFORM_URL: str = f"https://{PLATFORM}.api.riotgames.com"

# Regional routing (Match-V5, Account-V1) — routing continental
REGION: str = "europe"
REGION_URL: str = f"https://{REGION}.api.riotgames.com"

# ── Queue IDs ────────────────────────────────────────────────────────────────
RANKED_SOLO_QUEUE: int = 420    # Ranked Solo/Duo
RANKED_FLEX_QUEUE: int = 440    # Ranked Flex

# ── Crawler ──────────────────────────────────────────────────────────────────
MAX_MATCHES_PER_SUMMONER: int = 20  # Nombre de matchs récupérés par joueur
MAX_RETRIES: int = 3                # Tentatives max avant abandon (erreurs 5xx)
INITIAL_BACKOFF_S: float = 1.0      # Backoff initial (doublé à chaque retry)
REQUEST_TIMEOUT_S: int = 15         # Timeout par requête HTTP
