"""Paires minimales — accord en genre et en nombre dans le groupe nominal.

Phénomène testé (à la BLiMP) : la phrase correcte et la phrase fautive ne
diffèrent que par UN mot, dont la forme porte l'accord :

* déterminant–nom ......... « une maison » / « *un maison »
* nom–adjectif épithète ... « une robe blanche » / « *une robe blanc »,
                            « des chats noirs » / « *des chats noir »
* adjectif attribut ....... « La porte est fermée. » / « *La porte est fermé. »

Chaque gabarit est un cadre de phrase avec une case « {} » ; on y insère soit
la forme correcte, soit la forme fautive. Tout le reste de la phrase est donc
identique caractère pour caractère (différence minimale garantie par
construction).

Équilibre des longueurs : le féminin et le pluriel allongent en général le
mot (« blanc » → « blanche », « noir » → « noirs », « un » → « une »). Pour
qu'un modèle qui préfère simplement les phrases courtes ne gagne pas
gratuitement, chaque gabarit alterne des paires où la bonne forme est la plus
courte (masculin ou singulier attendu, ex. « un manteau noir » / « *un manteau
noire ») et des paires où elle est la plus longue (féminin ou pluriel attendu,
ex. « une robe blanche » / « *une robe blanc »). Seul le gabarit
« det_defini » (le / la) donne des phrases de même longueur : c'est imposé
par le phénomène, les deux articles ayant deux lettres.

Les noms à double genre (un livre / une livre, le tour / la tour, le page /
la page, le poste / la poste…) et les adjectifs invariables en genre ou en
nombre là où l'on teste ce trait (rouge au genre, vieux ou frais au pluriel
masculin…) sont volontairement exclus. Les formes « bel » et « vieil »
(devant voyelle) sont évitées ; « cet » n'apparaît que dans « Cet arbre »,
où c'est justement la bonne forme.

Module autonome, bibliothèque standard uniquement, sortie déterministe.
"""

PHENOMENE = "accord_nominal"

# Gabarit -> liste de (cadre, forme correcte, forme fautive).
# Le cadre contient exactement une case « {} ».
GABARITS = {
    # Genre : article indéfini un / une devant le nom.
    "det_indef": [
        ("Mon frère a trouvé {} couteau dans l'herbe.", "un", "une"),
        ("Le pêcheur répare {} bateau sur la plage.", "un", "une"),
        ("Le soldat dort dans {} fauteuil.", "un", "une"),
        ("Le meunier habite {} maison près du moulin.", "une", "un"),
        ("Il a écrit {} lettre à son ami.", "une", "un"),
        ("{} fleur pousse au bord du chemin.", "Une", "Un"),
    ],
    # Genre : article défini le / la (longueurs égales, imposé par le phénomène).
    "det_defini": [
        ("Le vent souffle fort sur {} mer.", "la", "le"),
        ("{} lune éclaire le village endormi.", "La", "Le"),
        ("Les enfants courent vers {} château.", "le", "la"),
    ],
    # Genre : démonstratif ce / cet / cette.
    "det_demonstratif": [
        ("Regarde {} navire sur la rivière.", "ce", "cette"),
        ("J'aime beaucoup {} chanson.", "cette", "ce"),
        ("{} arbre a plus de cent ans.", "Cet", "Cette"),
        ("{} ville dort sous la neige.", "Cette", "Ce"),
        ("{} verger appartient à mon oncle.", "Ce", "Cette"),
        ("Nous connaissons bien {} rue.", "cette", "ce"),
    ],
    # Genre : possessif mon / ma, ton / ta, son / sa (noms à consonne initiale).
    "det_possessif": [
        ("Il a perdu {} clé dans l'escalier.", "sa", "son"),
        ("Elle cherche {} chapeau partout.", "son", "sa"),
        ("Je range {} chambre le samedi.", "ma", "mon"),
        ("Tu as oublié {} parapluie chez nous.", "ton", "ta"),
        ("{} mère prépare le dîner.", "Ma", "Mon"),
        ("{} père lit le journal.", "Mon", "Ma"),
    ],
    # Nombre : le déterminant s'accorde avec le nom au singulier ou au pluriel.
    "det_nombre": [
        ("Je connais {} légendes depuis l'enfance.", "ces", "cette"),
        ("Il lave {} verres après le repas.", "les", "le"),
        ("Nous attendons {} cousins ce soir.", "nos", "notre"),
        ("Le marchand vend {} fruits au marché.", "des", "un"),
        ("Les paysans aiment {} village.", "leur", "leurs"),
    ],
    # Déterminants quantifieurs : chaque, plusieurs, aucun(e), tout(e).
    "quantifieur": [
        ("Il travaille chaque {} aux champs.", "jour", "jours"),
        ("J'ai visité plusieurs {} en été.", "villes", "ville"),
        ("Je n'ai reçu {} réponse.", "aucune", "aucun"),
        ("Il n'a fait {} bruit.", "aucun", "aucune"),
        ("{} la famille dort encore.", "Toute", "Tout"),
        ("{} le village chante ce soir.", "Tout", "Toute"),
    ],
    # Genre : adjectif épithète placé après le nom.
    "epithete_genre": [
        ("La mariée porte une robe {}.", "blanche", "blanc"),
        ("Le vieillard porte un manteau {}.", "noir", "noire"),
        ("Le musicien joue une valse {}.", "ancienne", "ancien"),
        ("Le prince monte un cheval {}.", "blanc", "blanche"),
        ("Cette femme a une voix {}.", "douce", "doux"),
        ("Le marin a un regard {}.", "sérieux", "sérieuse"),
    ],
    # Genre : adjectif antéposé à féminin irrégulier (vieux, beau, nouveau, bon, long, gros).
    "epithete_anteposee": [
        ("Il vit dans un {} château.", "vieux", "vieille"),
        ("Ils ont acheté une {} ferme.", "vieille", "vieux"),
        ("Mon voisin a un {} jardin.", "beau", "belle"),
        ("La reine porte une {} couronne.", "belle", "beau"),
        ("Le village a construit un {} pont.", "nouveau", "nouvelle"),
        ("Elle attend une {} nouvelle.", "bonne", "bon"),
        ("Il a fait un {} voyage.", "long", "longue"),
        ("Le fermier élève une {} vache.", "grosse", "gros"),
    ],
    # Nombre : adjectif épithète au singulier ou au pluriel.
    "epithete_nombre": [
        ("J'ai vu des chats {} dans la cour.", "noirs", "noir"),
        ("L'enfant a les yeux {}.", "bleus", "bleu"),
        ("Le quartier a des rues {}.", "étroites", "étroite"),
        ("Les {} jours reviennent enfin.", "beaux", "beau"),
        ("Le chasseur a un chien {}.", "fidèle", "fidèles"),
        ("Il boit un café {}.", "chaud", "chauds"),
        ("Elle cueille une rose {}.", "rouge", "rouges"),
    ],
    # Genre : adjectif (ou participe) attribut du sujet.
    "attribut_genre": [
        ("La porte est {}.", "fermée", "fermé"),
        ("Le musée est {} le dimanche.", "ouvert", "ouverte"),
        ("La neige est {} ce matin.", "blanche", "blanc"),
        ("Le ciel est {} aujourd'hui.", "bleu", "bleue"),
        ("Ma tante semble {}.", "heureuse", "heureux"),
        ("Le pain est encore {}.", "frais", "fraîche"),
        ("Le loup paraît {}.", "cruel", "cruelle"),
        ("La princesse était très {}.", "belle", "beau"),
    ],
    # Nombre : adjectif (ou participe) attribut du sujet.
    "attribut_nombre": [
        ("Les fenêtres sont {}.", "ouvertes", "ouverte"),
        ("La rivière est {}.", "profonde", "profondes"),
        ("Les enfants sont {} ce soir.", "fatigués", "fatigué"),
        ("Le chevalier est {}.", "blessé", "blessés"),
        ("Les rues semblent {}.", "désertes", "déserte"),
        ("Le lac reste {}.", "calme", "calmes"),
        ("Les fleurs sont {} au printemps.", "belles", "belle"),
        ("Le vin est {}.", "bon", "bons"),
    ],
}


def generate() -> list:
    """Liste de dicts {"good": str, "bad": str, "phenomene": PHENOMENE, "gabarit": str}.

    L'ordre est fixe (ordre de déclaration des gabarits puis des cadres) :
    aucune part d'aléatoire.
    """
    paires = []
    for gabarit, cadres in GABARITS.items():
        for cadre, bonne, mauvaise in cadres:
            paires.append({
                "good": cadre.format(bonne),
                "bad": cadre.format(mauvaise),
                "phenomene": PHENOMENE,
                "gabarit": gabarit,
            })
    return paires


if __name__ == "__main__":
    # Affichage rapide : python evals/paires/accord_nominal.py
    _paires = generate()
    for _p in _paires:
        print(f"[{_p['gabarit']}] {_p['good']}  |  *{_p['bad']}")
    _courtes = sum(len(p["good"]) < len(p["bad"]) for p in _paires)
    print(f"{len(_paires)} paires, bonne phrase plus courte : "
          f"{100 * _courtes / len(_paires):.1f} %")
