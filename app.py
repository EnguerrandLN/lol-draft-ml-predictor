"""
app.py — Conseiller de draft League of Legends (modèle additif).

Lancement :
  streamlit run app.py

Le modèle est lu depuis data/additive_model.json (produit par
ml/train.py) et rechargé automatiquement s'il est réentraîné.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import dotenv_values

sys.path.insert(0, str(Path(__file__).parent))
from api.client import ApiKeyError, ApiUnavailableError, RiotApiClient
from config import DATA_DIR
from db.schema import init_db
from ml.additive_model import AdditiveDraftModel
from ml.display_names import load_champion_display
from ml.draft_data import ROLES, connect_read_only
from ml.personal import ChampionProfile, fetch_history, player_profile
from ml.recommend import DraftRecommender, DraftState

MODEL_PATH: Path = DATA_DIR / "additive_model.json"
ROLE_LABELS: dict[str, str] = {"TOP": "Top", "JUNGLE": "Jungle", "MIDDLE": "Mid", "BOTTOM": "ADC", "UTILITY": "Support"}
SIDE_LABELS: dict[str, str] = {"blue": "🔵 Bleu", "red": "🔴 Rouge"}

st.set_page_config(page_title="LoL Draft Advisor", page_icon="🎯", layout="wide")


# ── Chargement ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_recommender(model_mtime: float) -> DraftRecommender:
    """`model_mtime` fait partie de la clé de cache : un modèle réentraîné est rechargé."""
    return DraftRecommender(AdditiveDraftModel.load(MODEL_PATH))


@st.cache_data
def load_display(internal_names: tuple[tuple[int, str], ...]) -> dict[int, dict]:
    return load_champion_display(dict(internal_names))


if not MODEL_PATH.exists():
    st.error(f"Modèle introuvable : {MODEL_PATH}. Lance d'abord `python ml/train.py`.")
    st.stop()

rec = load_recommender(MODEL_PATH.stat().st_mtime)
display = load_display(tuple(sorted(rec.names.items())))


def name(cid: int) -> str:
    return display[cid]["name"]


champ_ids: list[int] = sorted(display, key=lambda c: name(c).lower())


# ── Barre latérale ────────────────────────────────────────────────────────────

meta = rec.model.meta
with st.sidebar:
    st.header("Modèle")
    start, end = (pd.to_datetime(t, unit="ms").date() for t in meta["date_range"])
    test = meta["test_results"]["Modèle retenu"]
    st.caption(
        f"**{meta['n_matches']:,}** matchs Ranked Solo · {start} → {end}  \n"
        f"Sur la semaine la plus récente (jamais vue à l'entraînement) : "
        f"**{test['accuracy']:.1%}** de bonnes prédictions."
    )

    st.header("Paramètres")
    min_games = st.slider(
        "Matchs minimum au rôle", 0, 1000, 50, 10,
        help="Masque les champions trop rarement joués à ce rôle : leur estimation est peu fiable.",
    )
    risk = st.slider(
        "Prudence face aux counters", 0.0, 1.0, 0.0, 0.1,
        help="0 : classer sur la probabilité attendue.  \n"
             "1 : classer sur la probabilité si ton vis-à-vis choisit ton pire matchup.  \n"
             "Monte-la quand tu picks avant ton vis-à-vis (blind pick).",
    )
    top_n = st.slider("Nombre de suggestions", 5, 50, 15)


# ── Profil joueur ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_profile(puuid: str, model_mtime: float, n_player_matches: int) -> dict[int, ChampionProfile]:
    """Clé de cache : joueur, version du modèle et nombre de ses matchs en base."""
    return player_profile(rec.model, puuid)


def count_player_matches(puuid: str) -> int:
    conn = connect_read_only()
    n = conn.execute("SELECT COUNT(*) FROM participants WHERE puuid = ?", (puuid,)).fetchone()[0]
    conn.close()
    return n


def import_history(riot_id: str, limit: int) -> None:
    """Résout le Riot ID, récupère l'historique et la maîtrise, et les garde en session."""
    api_key = (dotenv_values(Path(__file__).parent / ".env").get("RIOT_API_KEY") or "").strip()
    if not api_key.startswith("RGAPI-"):
        st.error("Clé API absente du fichier .env (RIOT_API_KEY=RGAPI-...).")
        return
    game_name, _, tag_line = riot_id.partition("#")
    if not game_name or not tag_line:
        st.error("Format attendu : Pseudo#TAG")
        return

    client = RiotApiClient(api_key=api_key)
    try:
        account = client.get_account_by_riot_id(game_name.strip(), tag_line.strip())
        if not account:
            st.error(f"Riot ID introuvable : {riot_id}")
            return
        bar = st.progress(0.0, text="Récupération de l'historique...")
        conn = init_db()
        new = fetch_history(
            conn, client, account["puuid"], limit,
            progress=lambda done, total: bar.progress(done / total, text=f"Parties : {done}/{total}"),
        )
        conn.close()
        masteries = client.get_champion_masteries(account["puuid"])
    except ApiKeyError:
        st.error("Clé API refusée (expirée ?). Mets à jour .env puis réessaie.")
        return
    except ApiUnavailableError as exc:
        st.error(f"API Riot indisponible : {exc}")
        return

    bar.empty()
    st.session_state.player = {
        "riot_id": riot_id,
        "puuid": account["puuid"],
        "masteries": {m["championId"]: m["championPoints"] for m in masteries},
    }
    st.toast(f"{new} nouvelle(s) partie(s) importée(s).")


with st.sidebar:
    st.header("Ton profil")
    with st.form("profile"):
        riot_id = st.text_input("Riot ID", placeholder="Pseudo#TAG")
        limit = st.number_input("Parties à analyser", 20, 500, 100, 20,
                                help="Environ une requête API par partie absente de la base "
                                     "(quota partagé avec le crawler : compte ~2 min pour 100 parties).")
        submitted = st.form_submit_button("Charger mon historique", width="stretch")
    if submitted and riot_id:
        import_history(riot_id, int(limit))

player = st.session_state.get("player")
profiles: dict[int, ChampionProfile] = {}
if player:
    profiles = load_profile(player["puuid"], MODEL_PATH.stat().st_mtime, count_player_matches(player["puuid"]))
    with st.sidebar:
        games = sum(p.games for p in profiles.values())
        if games:
            wins = sum(p.wins for p in profiles.values())
            expected = sum(p.expected_wins for p in profiles.values())
            st.caption(
                f"**{player['riot_id']}** · {games} parties analysées  \n"
                f"Victoires : **{wins / games:.1%}**, contre {expected / games:.1%} attendus d'après tes drafts."
            )
        else:
            st.caption(f"**{player['riot_id']}** · aucune partie Ranked Solo exploitable.")


# ── Draft ─────────────────────────────────────────────────────────────────────

st.title("🎯 LoL Draft Advisor")

col_role, col_side = st.columns([3, 2])
role: str = col_role.segmented_control(
    "Rôle à pourvoir", ROLES, format_func=ROLE_LABELS.get, default="TOP", key="role"
) or "TOP"
side = col_side.segmented_control(
    "Ton côté (optionnel)", list(SIDE_LABELS), format_func=SIDE_LABELS.get, key="side"
)


def champion_select(label: str, key: str, disabled: bool = False):
    return st.selectbox(
        label, [None] + champ_ids, key=key, disabled=disabled,
        format_func=lambda c: "—" if c is None else name(c),
    )


col_ally, col_enemy = st.columns(2)
with col_ally:
    st.subheader("Ton équipe")
    ally = {
        r: champion_select(f"{ROLE_LABELS[r]}{' · à pourvoir' if r == role else ''}", f"ally_{r}", disabled=(r == role))
        for r in ROLES
    }
with col_enemy:
    st.subheader("Équipe adverse")
    enemy = {r: champion_select(ROLE_LABELS[r], f"enemy_{r}") for r in ROLES}

col_bans, col_pool = st.columns(2)
bans = col_bans.multiselect("Bans", champ_ids, format_func=name, max_selections=10)
pool = col_pool.multiselect(
    "Mon pool (optionnel)", champ_ids, format_func=name,
    help="Ne proposer que les champions que tu sais jouer.",
)
own_pool, min_own_games = False, 0
if profiles:
    with col_pool:
        own_pool = st.radio(
            "Champions proposés", ["Tous", "Seulement ceux que je joue"], horizontal=True,
            help="« Tous » permet de découvrir un champion fort dans cette situation : ton historique "
                 "s'affiche en colonnes et ajuste les champions que tu joues déjà. Un champion jamais "
                 "joué n'a pas de bonus perso, mais les premières parties sur un nouveau champion "
                 "coûtent en général quelques points (non mesuré ici).",
        ) == "Seulement ceux que je joue"
        if own_pool:
            min_own_games = st.slider("…avec au moins N parties à ce rôle", 1, 20, 3)

ally_picks = {r: c for r, c in ally.items() if c is not None and r != role}
enemy_picks = {r: c for r, c in enemy.items() if c is not None}
chosen = list(ally_picks.values()) + list(enemy_picks.values()) + bans
duplicates = sorted({name(c) for c in chosen if chosen.count(c) > 1})
if duplicates:
    st.error(f"Champion(s) sélectionné(s) plusieurs fois : {', '.join(duplicates)}.")
    st.stop()

state = DraftState(ally=ally_picks, enemy=enemy_picks, bans=set(bans), ally_side=side)


# ── Recommandations ───────────────────────────────────────────────────────────

st.divider()
candidate_pool = set(pool)
if own_pool:
    candidate_pool |= {c for c, p in profiles.items() if p.games_by_role.get(role, 0) >= min_own_games}
results = rec.recommend(
    state, role,
    pool=candidate_pool if (pool or own_pool) else None,
    # Tes propres champions restent proposés même s'ils sont rares à ce rôle :
    # l'atténuation du modèle ramène déjà leur effet global vers 0.
    min_games=0 if own_pool else min_games,
    risk=risk,
    personal={c: p.effect for c, p in profiles.items()} or None,
)

n_known = len(ally_picks) + len(enemy_picks)
col_metric, col_info = st.columns([1, 3])
col_metric.metric(
    "Victoire avec un pick moyen", f"{rec.draft_win_prob(state):.1%}",
    help="Probabilité de victoire de la draft actuelle si tu prends un champion moyen à ce rôle. "
         "Les slots encore vides sont estimés d'après ce qui se joue habituellement.",
)
col_info.caption(
    f"{n_known}/9 autres champions connus · {len(bans)} ban(s). "
    + ("Ton vis-à-vis est connu : « Si contré » = « Victoire »." if role in enemy_picks
       else "Ton vis-à-vis n'est pas encore choisi : « Si contré » montre ton exposition aux counters.")
)

if not results:
    st.info("Aucun candidat : baisse le seuil de matchs minimum ou élargis ton pool.")
    st.stop()

st.subheader(f"Meilleurs picks · {ROLE_LABELS[role]}")
def result_row(r) -> dict:
    row = {
        "icon": display[r.champion_id]["icon"],
        "Champion": name(r.champion_id),
        "Victoire": r.win_prob * 100,
        "vs pick moyen": r.vs_average * 100,
        "Si contré": r.win_prob_if_countered * 100,
        "Pire matchup": name(r.worst_response_id) if r.worst_response_id is not None else "—",
        "Matchs au rôle": r.games,
    }
    if profiles:
        prof = profiles.get(r.champion_id)
        row["Tes parties"] = prof.games_by_role.get(role, 0) if prof else 0
        row["Ton winrate"] = prof.wins / prof.games * 100 if prof else float("nan")  # Vide si jamais joué
        row["Bonus perso"] = r.personal_effect * 100
        row["Maîtrise"] = player["masteries"].get(r.champion_id, 0)
    return row


table = pd.DataFrame([result_row(r) for r in results[:top_n]])
st.dataframe(
    table,
    hide_index=True,
    width="stretch",
    height=min(38 + 35 * len(table), 720),
    column_config={
        "icon": st.column_config.ImageColumn("", width="small"),
        "Victoire": st.column_config.ProgressColumn("Victoire", format="%.1f %%", min_value=40, max_value=60),
        "vs pick moyen": st.column_config.NumberColumn("vs pick moyen", format="%+.1f pts"),
        "Si contré": st.column_config.NumberColumn("Si contré", format="%.1f %%"),
        "Matchs au rôle": st.column_config.NumberColumn("Matchs au rôle", format="%d"),
        "Tes parties": st.column_config.NumberColumn("Tes parties", format="%d",
                                                     help="Tes parties sur ce champion à ce rôle."),
        "Ton winrate": st.column_config.NumberColumn("Ton winrate", format="%.0f %%",
                                                     help="Tous rôles confondus, sur les parties analysées."),
        "Bonus perso": st.column_config.NumberColumn(
            "Bonus perso", format="%+.1f pts",
            help="Ce que ton historique ajoute à la probabilité : tes victoires au-delà de ce que "
                 "tes drafts laissaient attendre, fortement atténuées tant que tu as peu de parties."),
        "Maîtrise": st.column_config.NumberColumn("Maîtrise", format="%d"),
    },
)

with st.expander("Comment lire ces chiffres"):
    st.markdown(
        "- **Victoire** : probabilité de victoire de ton équipe si tu prends ce champion, "
        "les picks encore inconnus étant estimés d'après ce qui se joue habituellement.\n"
        "- **vs pick moyen** : ce que ce choix ajoute par rapport à un champion moyen à ce rôle.\n"
        "- **Si contré** : la même probabilité si ton vis-à-vis choisit, parmi ses picks courants, "
        "ton pire matchup (**Pire matchup**).\n"
        "- **Matchs au rôle** : nombre de parties réelles sur lesquelles repose l'estimation.\n"
        "- **Bonus perso** (si ton profil est chargé) : tes victoires au-delà de ce que tes drafts "
        "laissaient attendre sur ce champion. Il est volontairement prudent : l'écart réel entre "
        "joueurs sur un même champion est de quelques points, et il faut des centaines de "
        "parties pour le distinguer du hasard.\n\n"
        "Les écarts sont de quelques points : la draft compte, mais moins que l'exécution. "
        "Les winrates d'un champion sont mesurés sur ceux qui le jouent : un pick de niche "
        "(ex. un mage en ADC) est surtout joué par des spécialistes, et son score ne s'applique "
        "pas forcément à toi. Le filtre **Mon pool** est là pour ça."
    )
