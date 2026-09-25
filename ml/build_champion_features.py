import json
import urllib.request
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR

def build_features():
    # 1. Fetch latest Data Dragon version
    with urllib.request.urlopen("https://ddragon.leagueoflegends.com/api/versions.json") as res:
        versions = json.loads(res.read().decode())
        latest = versions[0]
        
    print(f"Fetching Data Dragon version {latest}...")
    
    # 2. Fetch Champions JSON
    url = f"https://ddragon.leagueoflegends.com/cdn/{latest}/data/en_US/champion.json"
    with urllib.request.urlopen(url) as res:
        data = json.loads(res.read().decode())["data"]
        
    features_dict = {}
    
    all_tags = ["Fighter", "Tank", "Mage", "Assassin", "Marksman", "Support"]
    
    for champ_name, champ_data in data.items():
        riot_id = int(champ_data["key"])
        
        # 3. Extract Features
        # stats
        attackrange = champ_data["stats"].get("attackrange", 125)
        # Rakan et Lillia sont mêlés et ont 300 de portée. Le tireur le plus court est Urgot avec 350.
        is_ranged = 1.0 if attackrange > 300 else 0.0
        
        # info
        info = champ_data["info"]
        att = info.get("attack", 0) / 10.0
        dfn = info.get("defense", 0) / 10.0
        mag = info.get("magic", 0) / 10.0
        dif = info.get("difficulty", 0) / 10.0
        
        # tags
        tags = champ_data.get("tags", [])
        tags_onehot = [1.0 if tag in tags else 0.0 for tag in all_tags]
        
        # Feature vector
        vec = [is_ranged, att, dfn, mag, dif] + tags_onehot
        features_dict[str(riot_id)] = vec

    out_path = DATA_DIR / "champion_features.json"
    with open(out_path, "w") as f:
        json.dump(features_dict, f, indent=4)
        
    print(f"Sauvegardé : {len(features_dict)} champions dans {out_path}.")
    print(f"Taille du vecteur : {len(vec)} features.")

if __name__ == "__main__":
    build_features()
