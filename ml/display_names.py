"""
display_names.py — Noms affichables et icônes des champions (Data Dragon).

La base ne contient que les noms internes de Riot (« MonkeyKing », « KSante »).
Data Dragon (CDN public de Riot, sans clé API) fournit les vrais noms et les
icônes. Le résultat est mis en cache dans data/champion_display.json ; sans
réseau, on retombe sur les noms internes.
"""
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR

logger = logging.getLogger(__name__)

DDRAGON = "https://ddragon.leagueoflegends.com"
CACHE_PATH: Path = DATA_DIR / "champion_display.json"


def _download(locale: str) -> Optional[dict]:
    try:
        version = requests.get(f"{DDRAGON}/api/versions.json", timeout=10).json()[0]
        data = requests.get(f"{DDRAGON}/cdn/{version}/data/{locale}/champion.json", timeout=10).json()["data"]
    except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
        logger.warning("Data Dragon indisponible (%s) : noms internes utilisés.", exc)
        return None
    return {
        "version": version,
        "champions": {
            champ["key"]: {"name": champ["name"], "icon": f"{DDRAGON}/cdn/{version}/img/champion/{champ_id}.png"}
            for champ_id, champ in data.items()
        },
    }


def load_champion_display(fallback_names: dict[int, str], locale: str = "fr_FR",
                          refresh: bool = False) -> dict[int, dict[str, Optional[str]]]:
    """
    Returns:
        champion_id → {"name": nom affichable, "icon": URL de l'icône ou None}.
        Couvre tous les ids de `fallback_names` (noms internes en secours).
    """
    cache = None
    if CACHE_PATH.exists() and not refresh:
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    # Champion inconnu du cache (nouveau champion) → on rafraîchit
    if cache is None or any(str(cid) not in cache["champions"] for cid in fallback_names):
        fresh = _download(locale)
        if fresh is not None:
            cache = fresh
            CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    known = cache["champions"] if cache else {}
    return {
        cid: known.get(str(cid), {"name": name, "icon": None})
        for cid, name in fallback_names.items()
    }
