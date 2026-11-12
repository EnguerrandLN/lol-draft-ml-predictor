"""
api/lcu.py — Lecture de la sélection de champions en cours dans le client League of Legends.

Le client du jeu expose sur la machine locale une API HTTPS (dite « LCU »),
celle qu'utilise sa propre interface. Elle donne en direct l'état de la
sélection de champions : picks, bans, rôle et côté du joueur. On la lit en
lecture seule, sur 127.0.0.1 uniquement.

Accès : le client écrit à son démarrage un fichier `lockfile` dans son dossier
d'installation, au format `LeagueClient:<pid>:<port>:<mot de passe>:<protocole>`.
Le port et le mot de passe changent à chaque lancement ; le mot de passe n'est
jamais affiché ni journalisé. Si le fichier n'est pas à l'emplacement habituel,
on lit la ligne de commande du processus du client (Windows), ou la variable
d'environnement LOL_LOCKFILE.

Le certificat du client est auto-signé : la vérification TLS est désactivée
pour cet appel local.

Cette API n'est pas une API publique de Riot : usage personnel ; une diffusion
de l'outil demanderait de l'enregistrer auprès de Riot.
"""
import os
import re
import subprocess
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

LOCKFILE_ENV: str = "LOL_LOCKFILE"
DEFAULT_LOCKFILES: tuple[Path, ...] = (
    Path(r"C:\Riot Games\League of Legends\lockfile"),
    Path(r"D:\Riot Games\League of Legends\lockfile"),
    Path("/Applications/League of Legends.app/Contents/LoL/lockfile"),
)
CHAMP_SELECT_ENDPOINT: str = "/lol-champ-select/v1/session"
TIMEOUT_S: float = 2.0
PROCESS_PROBE_INTERVAL_S: float = 15.0   # Lire la liste des processus est lent : pas à chaque interrogation
_process_probe: list = [float("-inf"), None]   # (instant du dernier examen, identifiants trouvés)

NO_CLIENT: str = "no_client"                # Client du jeu fermé ou introuvable
NO_CHAMP_SELECT: str = "no_champ_select"    # Client ouvert, pas de sélection en cours
OK: str = "ok"


@dataclass(frozen=True)
class Credentials:
    port: int
    token: str
    protocol: str = "https"

    @property
    def base_url(self) -> str:
        return f"{self.protocol}://127.0.0.1:{self.port}"


def parse_lockfile(text: str) -> Optional[Credentials]:
    """`LeagueClient:<pid>:<port>:<mot de passe>:<protocole>` → identifiants."""
    parts = text.strip().split(":")
    if len(parts) != 5 or not parts[2].isdigit():
        return None
    return Credentials(port=int(parts[2]), token=parts[3], protocol=parts[4])


def parse_command_line(command_line: str) -> Optional[Credentials]:
    """Identifiants lus dans la ligne de commande de LeagueClientUx.exe."""
    port = re.search(r"--app-port=(\d+)", command_line)
    token = re.search(r"--remoting-auth-token=([\w-]+)", command_line)
    return Credentials(port=int(port.group(1)), token=token.group(1)) if port and token else None


def _credentials_from_process() -> Optional[Credentials]:
    if os.name != "nt":
        return None
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"name='LeagueClientUx.exe'\").CommandLine"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_command_line(out)


def find_credentials() -> Optional[Credentials]:
    """Identifiants du client en cours d'exécution, ou None s'il est fermé."""
    candidates = ([Path(os.environ[LOCKFILE_ENV])] if os.environ.get(LOCKFILE_ENV) else []) + list(DEFAULT_LOCKFILES)
    for path in candidates:
        try:
            creds = parse_lockfile(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        if creds:
            return creds
    if time.monotonic() - _process_probe[0] >= PROCESS_PROBE_INTERVAL_S:
        _process_probe[:] = [time.monotonic(), _credentials_from_process()]
    return _process_probe[1]


def champ_select_session(creds: Optional[Credentials] = None) -> tuple[str, Optional[dict]]:
    """
    État de la sélection de champions en cours.

    Returns:
        (statut, session) : statut parmi OK, NO_CHAMP_SELECT, NO_CLIENT ; la
        session (JSON du client) n'est fournie qu'avec OK.
    """
    creds = creds or find_credentials()
    if creds is None:
        return NO_CLIENT, None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # Certificat auto-signé du client, appel local
            response = requests.get(creds.base_url + CHAMP_SELECT_ENDPOINT, auth=("riot", creds.token),
                                    verify=False, timeout=TIMEOUT_S)
    except requests.exceptions.RequestException:
        return NO_CLIENT, None                # Lockfile périmé ou client en cours de fermeture
    if response.status_code != 200:
        return NO_CHAMP_SELECT, None
    try:
        return OK, response.json()
    except ValueError:
        return NO_CHAMP_SELECT, None
