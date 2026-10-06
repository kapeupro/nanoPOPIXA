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

Deux gabarits empêchent de résoudre le jeu par de simples indices de surface :

* « genre_nom » : le déterminant ne porte PAS le genre (l', des, leur,
  ou « mon » devant un nom féminin à voyelle : « Mon école est très
  grande. »). Seul le genre lexical du nom décide de la forme de l'adjectif ;
  certaines phrases ajoutent même un leurre de l'autre genre (« Ma sœur
  porte des gants gris. », « Cette année, l'hiver fut très froid. »).
* « accord_distance » : un complément du nom de genre ou de nombre opposé
  s'intercale entre le nom et l'attribut (« Le vin de ces collines est
  bon. », « L'histoire de ce vieux roi est vraie. ») ; l'accord ne se joue
  plus dans une fenêtre de deux ou trois mots.

Les adjectifs attributs sont des adjectifs purs, sauf « fermée » (exemple de
référence du phénomène) et « ouvert », adjectifs lexicalisés : les accords
de participes passés après « être » relèvent de participe_passe.py.

Équilibre des longueurs : le féminin et le pluriel allongent en général le
mot (« blanc » → « blanche », « noir » → « noirs », « un » → « une »). Pour
qu'un modèle qui préfère simplement les phrases courtes ne gagne pas
gratuitement, chaque gabarit contient autant de paires où la bonne forme est
la plus courte (masculin ou singulier attendu, ex. « un manteau noir » /
« *un manteau noire ») que de paires où elle est la plus longue (féminin ou
pluriel attendu, ex. « une robe blanche » / « *une robe blanc »). Seul le
gabarit « det_defini » (le / la) donne des phrases de même longueur : c'est
imposé par le phénomène, les deux articles ayant deux lettres.

Accord audible ou muet : environ un tiers des paires opposent des formes
homophones (« noir » / « noire », « leur » / « leurs », « cet » / « cette ») ;
l'accord n'y est qu'orthographique et son apprentissage dépend beaucoup de la
tokenisation des suffixes -e et -s. Chaque cadre porte donc une étiquette
« audible » ; accord_audible() renvoie {phrase correcte: bool} pour publier
les deux sous-scores, sans changer le format des paires.

Les noms à double genre (un livre / une livre, le tour / la tour, le page /
la page, le poste / la poste…) et les adjectifs invariables en genre ou en
nombre là où l'on teste ce trait (rouge au genre, vieux ou frais au pluriel
masculin…) sont volontairement exclus. Les formes « bel », « vieil » et
« nouvel » (devant voyelle) sont évitées ; « cet » n'apparaît que dans
« Cet arbre », où c'est justement la bonne forme.

Module autonome, bibliothèque standard uniquement, sortie déterministe.
"""

PHENOMENE = "accord_nominal"

# Étiquettes d'audibilité : la forme correcte et la forme fautive se
# prononcent-elles différemment (AUDIBLE) ou pareil (MUET) ?
AUDIBLE, MUET = True, False

# Gabarit -> liste de (cadre, forme correcte, forme fautive, audibilité).
# Le cadre contient exactement une case « {} ».
GABARITS = {
    # Genre : article indéfini un / une devant le nom.
    "det_indef": [
        ("Mon frère a trouvé {} couteau dans l'herbe.", "un", "une", AUDIBLE),
        ("Le pêcheur répare {} bateau sur la plage.", "un", "une", AUDIBLE),
        ("Le meunier habite {} maison près du moulin.", "une", "un", AUDIBLE),
        ("{} fleur pousse au bord du chemin.", "Une", "Un", AUDIBLE),
    ],
    # Genre : article défini le / la (longueurs égales, imposé par le phénomène).
    "det_defini": [
        ("Mon cousin n'a jamais vu {} mer.", "la", "le", AUDIBLE),
        ("{} lune éclaire le village endormi.", "La", "Le", AUDIBLE),
        ("Les soldats marchent vers {} château.", "le", "la", AUDIBLE),
    ],
    # Genre : démonstratif ce / cet / cette.
    "det_demonstratif": [
        ("Regarde {} cygne sur la rivière.", "ce", "cette", AUDIBLE),
        ("J'aime beaucoup {} chanson.", "cette", "ce", AUDIBLE),
        ("{} arbre a plus de cent ans.", "Cet", "Cette", MUET),
        ("{} ville possède un grand port.", "Cette", "Ce", AUDIBLE),
    ],
    # Genre : possessif mon / ma, ton / ta, son / sa (noms à consonne initiale).
    "det_possessif": [
        ("Il a perdu {} clé dans l'escalier.", "sa", "son", AUDIBLE),
        ("Elle cherche {} chapeau partout.", "son", "sa", AUDIBLE),
        ("Je range {} chambre le samedi.", "ma", "mon", AUDIBLE),
        ("Tu as oublié {} parapluie chez nous.", "ton", "ta", AUDIBLE),
    ],
    # Nombre : le déterminant s'accorde avec le nom au singulier ou au pluriel
    # (pluriel attendu dans 3 paires, singulier dans 3).
    "det_nombre": [
        ("Je connais {} légendes depuis l'enfance.", "ces", "cette", AUDIBLE),
        ("Il lave {} verres après le repas.", "les", "le", AUDIBLE),
        ("Le marchand vend {} fruits au marché.", "des", "un", AUDIBLE),
        ("Les paysans aiment {} village.", "leur", "leurs", MUET),
        ("Le cuisinier sort {} gâteau du four.", "le", "les", AUDIBLE),
        ("Nous prendrons {} voiture demain matin.", "notre", "nos", AUDIBLE),
    ],
    # Déterminants quantifieurs : chaque, plusieurs, aucun(e), tout(e).
    "quantifieur": [
        ("Il travaille chaque {} aux champs.", "jour", "jours", MUET),
        ("J'ai visité plusieurs {} pendant le voyage.", "villes", "ville", MUET),
        ("Je n'ai reçu {} réponse.", "aucune", "aucun", AUDIBLE),
        ("Il n'a fait {} bruit.", "aucun", "aucune", AUDIBLE),
        ("{} la famille dort encore.", "Toute", "Tout", AUDIBLE),
        ("{} le peuple acclame le roi.", "Tout", "Toute", AUDIBLE),
    ],
    # Genre : adjectif épithète placé après le nom.
    "epithete_genre": [
        ("La mariée porte une robe {}.", "blanche", "blanc", AUDIBLE),
        ("Le vieillard porte un manteau {}.", "noir", "noire", MUET),
        ("Le musicien joue une valse {}.", "ancienne", "ancien", AUDIBLE),
        ("Le peintre a un atelier {}.", "lumineux", "lumineuse", AUDIBLE),
        ("Cette femme a une voix {}.", "douce", "doux", AUDIBLE),
        ("Le marin a un regard {}.", "sérieux", "sérieuse", AUDIBLE),
    ],
    # Genre : adjectif antéposé à féminin irrégulier (vieux, beau, nouveau, bon, gros).
    "epithete_anteposee": [
        ("Il vit dans un {} château.", "vieux", "vieille", AUDIBLE),
        ("Ils ont acheté une {} ferme.", "vieille", "vieux", AUDIBLE),
        ("Mon oncle possède un {} cheval.", "beau", "belle", AUDIBLE),
        ("La commune a construit un {} pont.", "nouveau", "nouvelle", AUDIBLE),
        ("Elle attend une {} nouvelle.", "bonne", "bon", AUDIBLE),
        ("Le fermier élève une {} vache.", "grosse", "gros", AUDIBLE),
    ],
    # Nombre : adjectif épithète au singulier ou au pluriel.
    "epithete_nombre": [
        ("J'ai vu des chats {} dans la cour.", "noirs", "noir", MUET),
        ("L'enfant a les yeux {}.", "bleus", "bleu", MUET),
        ("Avec le mois de mai, les {} jours reviennent enfin.", "beaux", "beau", MUET),
        ("Le chasseur a un chien {}.", "fidèle", "fidèles", MUET),
        ("Chaque matin, le facteur boit un café {}.", "chaud", "chauds", MUET),
        ("Au fond du parc, elle cueille une rose {}.", "rouge", "rouges", MUET),
    ],
    # Genre : adjectif attribut du sujet.
    "attribut_genre": [
        ("La porte est {}.", "fermée", "fermé", MUET),
        ("Le musée est {} le dimanche.", "ouvert", "ouverte", AUDIBLE),
        ("Ma tante semble {} depuis son mariage.", "heureuse", "heureux", AUDIBLE),
        ("À midi, le pain est encore {}.", "frais", "fraîche", AUDIBLE),
        ("Dans la fable, le loup paraît {}.", "cruel", "cruelle", MUET),
        ("La princesse était très {}.", "jalouse", "jaloux", AUDIBLE),
    ],
    # Nombre : adjectif attribut du sujet.
    "attribut_nombre": [
        ("Les routes sont {} en hiver.", "dangereuses", "dangereuse", MUET),
        ("La forêt est {} et silencieuse.", "sombre", "sombres", MUET),
        ("Le chevalier est {}.", "fier", "fiers", MUET),
        ("En novembre, les plages sont {}.", "désertes", "déserte", MUET),
        ("Le lac reste {} malgré le vent.", "calme", "calmes", MUET),
        ("Les journées sont {} en été.", "longues", "longue", MUET),
    ],
    # Genre porté par le nom seul : déterminant non marqué en genre (l', des,
    # leur, « mon » devant voyelle), parfois avec un leurre de l'autre genre.
    "genre_nom": [
        ("Mon école est très {}.", "grande", "grand", AUDIBLE),
        ("Cette année, l'hiver fut très {}.", "froid", "froide", AUDIBLE),
        ("Ma sœur porte des gants {}.", "gris", "grises", AUDIBLE),
        ("Il a acheté des chaussures {}.", "neuves", "neufs", AUDIBLE),
        ("L'armoire du grenier est très {}.", "lourde", "lourd", AUDIBLE),
        ("Leur fils est très {}.", "courageux", "courageuse", AUDIBLE),
    ],
    # Accord à distance : complément du nom intercalé, de genre ou de nombre opposé.
    "accord_distance": [
        ("Le panier de ma grand-mère est {}.", "léger", "légère", AUDIBLE),
        ("La cabane du berger est {}.", "petite", "petit", AUDIBLE),
        ("Les cerises du jardin sont {}.", "mûres", "mûrs", MUET),
        ("Le chef des brigands était {}.", "méchant", "méchants", MUET),
        ("Le vin de ces collines est {}.", "bon", "bons", MUET),
        ("L'histoire de ce vieux roi est {}.", "vraie", "vrai", MUET),
    ],
}


def generate() -> list:
    """Liste de dicts {"good": str, "bad": str, "phenomene": PHENOMENE, "gabarit": str}.

    L'ordre est fixe (ordre de déclaration des gabarits puis des cadres) :
    aucune part d'aléatoire.
    """
    paires = []
    for gabarit, cadres in GABARITS.items():
        for cadre, bonne, mauvaise, _audible in cadres:
            paires.append({
                "good": cadre.format(bonne),
                "bad": cadre.format(mauvaise),
                "phenomene": PHENOMENE,
                "gabarit": gabarit,
            })
    return paires


def accord_audible() -> dict:
    """{phrase correcte: True si l'accord s'entend à l'oral, False s'il est muet}.

    Permet de calculer deux sous-scores (accord audible / accord purement
    orthographique) sans ajouter de clé aux paires de generate().
    """
    return {
        cadre.format(bonne): audible
        for cadres in GABARITS.values()
        for cadre, bonne, _mauvaise, audible in cadres
    }


if __name__ == "__main__":
    # Affichage rapide : python evals/paires/accord_nominal.py
    _paires = generate()
    _audible = accord_audible()
    for _p in _paires:
        _tag = "oral" if _audible[_p["good"]] else "muet"
        print(f"[{_p['gabarit']}|{_tag}] {_p['good']}  |  *{_p['bad']}")
    _courtes = sum(len(p["good"]) < len(p["bad"]) for p in _paires)
    _muettes = sum(not v for v in _audible.values())
    print(f"{len(_paires)} paires, bonne phrase plus courte : "
          f"{100 * _courtes / len(_paires):.1f} %, accord muet : {_muettes}")
