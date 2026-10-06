"""Paires minimales : élision et h aspiré.

Phénomène testé (à la BLiMP) : devant une voyelle ou un h muet, les mots
grammaticaux brefs le, la, de, que (lorsque, jusque), je, me, te, se, ne, ce
perdent leur voyelle finale, remplacée par une apostrophe ; « si » ne s'élide
que devant « il » et « ils ». Devant un h aspiré, en revanche, l'élision est
interdite. La phrase correcte et la phrase fautive ne diffèrent que par cette
élision :

* élision obligatoire (la bonne phrase est la forme élidée) :
      « l'arbre » / « *le arbre », « l'eau » / « *la eau »,
      « l'homme » / « *le homme » (h muet), « d'eau » / « *de eau »,
      « qu'il » / « *que il », « jusqu'au » / « *jusque au »,
      « j'aime » / « *je aime », « s'il » / « *si il »,
      « s'habille » / « *se habille », « c'est » / « *ce est »
* élision interdite (la bonne phrase est la forme pleine) :
      h aspiré : « le héros » / « *l'héros », « la hache » / « *l'hache »,
      « le hibou » / « *l'hibou », « de honte » / « *d'honte »,
      « je hais » / « *j'hais », « ne hurle » / « *n'hurle »,
      « se hérisse » / « *s'hérisse » ;
      « si » devant autre chose que « il(s) » : « si elle » / « *s'elle »,
      « si on » / « *s'on », « si un » / « *s'un ».

Construction : chaque entrée est (cadre, mot élidable, mot suivant). Le cadre
contient une case « {} » qui reçoit soit « mot suivant » (forme pleine), soit
la forme élidée (« le » → « l' », « lorsque » → « lorsqu' »...) collée au mot
suivant. Le reste de la phrase est donc identique caractère pour caractère
(différence minimale garantie par construction). La règle du gabarit
(ELISION ou DISJONCTION) dit laquelle des deux formes est la bonne. Plusieurs
gabarits opposent volontairement un h muet à un h aspiré du même champ
lexical (« l'héroïne » / « le héros », « n'hésite » / « ne hurle »,
« plein d'espoir » / « plein de haine ») : le modèle doit connaître le mot,
pas seulement sa première lettre.

Équilibre des longueurs : l'élision raccourcit toujours la phrase d'un
caractère (voyelle + espace remplacées par l'apostrophe). Un modèle qui
préfère les phrases courtes gagnerait donc toutes les paires « élision
obligatoire » et perdrait toutes les paires « h aspiré » : les deux familles
sont presque à parité (34 paires où la bonne phrase est la plus courte,
32 où elle est la plus longue, soit 51,5 % / 48,5 %). Aucune paire n'a deux
phrases de même longueur : c'est imposé par le phénomène.

Précautions linguistiques :
  - seuls des h aspirés et des h muets stables dans l'usage écrit sont
    retenus ; les mots à usage flottant sont exclus (hyène, handicap,
    hiatus, ouate, ouistiti...), ainsi que « onze » et « huit », dont
    l'élision (« l'onzième ») se lit encore chez les classiques ;
  - pas de « presque » ni de « quelque » (« presqu'entier »,
    « quelqu'autre » se rencontrent dans les textes anciens) ;
  - « s'elle » et « s'on » n'appartiennent qu'au moyen français : chez
    Molière comme aujourd'hui, seul « si » + elle / on / un est admis ;
  - aucun article masculin après « à » ou « de » dans les cadres : la forme
    fautive « *à le héros » cumulerait deux fautes (contraction et élision),
    et « au héros » / « *à l'héros » ne serait plus une différence minimale ;
  - aucune question (toutes les phrases finissent par un point) ; les autres
    apostrophes du cadre (« aujourd'hui », « l'année ») sont identiques dans
    les deux phrases.

Module autonome, bibliothèque standard uniquement, sortie déterministe.
"""

PHENOMENE = "elision"

# Règle d'un gabarit : quelle forme est la bonne.
ELISION = "elision"          # élision obligatoire : la forme élidée est correcte
DISJONCTION = "disjonction"  # élision interdite (h aspiré, « si » + elle/on/un)

# Gabarit -> (règle, liste de (cadre, mot élidable, mot suivant)).
# Le cadre contient exactement une case « {} ».
GABARITS = {
    # Article défini le / la devant une voyelle.
    "article_voyelle": (ELISION, [
        ("Le paysan plante {} au bout du champ.", "le", "arbre"),
        ("{} de la rivière est très froide.", "La", "eau"),
        ("{} joue seul dans la cour.", "Le", "enfant"),
        ("Le soleil se lève sur {}.", "la", "île"),
        ("{} gronde au-dessus de la ville.", "Le", "orage"),
    ]),
    # Article défini le / la devant un h muet.
    "article_h_muet": (ELISION, [
        ("{} marche lentement sur la route.", "Le", "homme"),
        ("Les vaches mangent {} du pré.", "la", "herbe"),
        ("{} arrive avec la neige et le froid.", "Le", "hiver"),
        ("Le vieux marin raconte {} de son naufrage.", "la", "histoire"),
        ("{} du roman meurt à la fin.", "La", "héroïne"),
        ("Le voyageur cherche {} de la gare.", "le", "hôtel"),
    ]),
    # Article défini le / la devant un h aspiré : pas d'élision.
    "article_h_aspire": (DISJONCTION, [
        ("{} revient enfin dans son village.", "Le", "héros"),
        ("Le bûcheron saisit {} sans un mot.", "la", "hache"),
        ("On entend {} au fond du bois.", "le", "hibou"),
        ("Les enfants jouent sous {}.", "le", "hêtre"),
        ("Le jardinier taille {} devant la maison.", "la", "haie"),
        ("{} dort au pied de la montagne.", "Le", "hameau"),
        ("{} fait bien les choses.", "Le", "hasard"),
        ("Le cuisinier prépare {} pour le dîner.", "le", "homard"),
        ("{} traverse le chemin sans bruit.", "Le", "hérisson"),
        ("Le fermier range le foin dans {}.", "le", "hangar"),
        ("{} secoue le petit bateau.", "La", "houle"),
        ("La princesse joue de {} au salon.", "la", "harpe"),
        ("Mon grand-père a mal à {}.", "la", "hanche"),
    ]),
    # Préposition de devant une voyelle ou un h muet.
    "de_elide": (ELISION, [
        ("Il boit un verre {} fraîche.", "de", "eau"),
        ("Elle a beaucoup {} au village.", "de", "amis"),
        ("Le jeune soldat est plein {}.", "de", "espoir"),
        ("Le roi porte une couronne {}.", "de", "or"),
    ]),
    # Préposition de devant un h aspiré : pas d'élision.
    "de_h_aspire": (DISJONCTION, [
        ("Il rougit {} devant son père.", "de", "honte"),
        ("Son cœur est plein {}.", "de", "haine"),
        ("Elle prépare un plat {} verts.", "de", "haricots"),
        ("Le pêcheur vend une caisse {}.", "de", "harengs"),
        ("Une branche {} décore la cheminée.", "de", "houx"),
    ]),
    # Que et ses composés (lorsque, jusque) devant une voyelle.
    "conjonction_que": (ELISION, [
        ("Je sais {} viendra demain.", "que", "il"),
        ("Il faut {} parte avant la nuit.", "que", "elle"),
        ("Le pauvre berger n'a {} seul ami.", "que", "un"),
        ("{} entra, tout le monde se tut.", "Lorsque", "il"),
        ("Nous avons dansé {} matin.", "jusque", "au"),
    ]),
    # Pronom sujet je devant une voyelle ou un h muet.
    "pronom_je": (ELISION, [
        ("{} la musique de mon pays.", "Je", "aime"),
        ("{} près de la gare depuis un an.", "Je", "habite"),
        ("{} la mer depuis ma chambre.", "Je", "entends"),
        ("Hier soir, {} vu ton frère au marché.", "je", "ai"),
    ]),
    # Pronom sujet je devant un verbe à h aspiré : pas d'élision.
    "je_h_aspire": (DISJONCTION, [
        ("{} le mensonge et la flatterie.", "Je", "hais"),
        ("{} les épaules sans répondre.", "Je", "hausse"),
        ("{} le persil pour la soupe.", "Je", "hache"),
        ("{} la table en passant.", "Je", "heurte"),
    ]),
    # Si s'élide devant il / ils.
    "si_il": (ELISION, [
        ("{} pleut demain, nous resterons ici.", "Si", "il"),
        ("Je ne sais pas {} dort encore.", "si", "il"),
        ("{} arrivent tôt, nous dînerons ensemble.", "Si", "ils"),
    ]),
    # Si ne s'élide pas devant elle, on, un.
    "si_sans_elision": (DISJONCTION, [
        ("{} vient, nous partirons à midi.", "Si", "elle"),
        ("Je me demande {} viendra ce soir.", "si", "elle"),
        ("{} frappe, ouvre la porte.", "Si", "on"),
        ("{} enfant pleure, il faut le consoler.", "Si", "un"),
    ]),
    # Pronoms conjoints me, te, se, le et négation ne devant voyelle ou h muet.
    "clitique_elide": (ELISION, [
        ("Elle {} vite le matin.", "se", "habille"),
        ("Ma mère {} depuis le jardin.", "me", "appelle"),
        ("Je {} devant la porte de l'église.", "te", "attends"),
        ("Le capitaine {} jamais devant le danger.", "ne", "hésite"),
        ("Je {} chanter dans la cuisine.", "le", "entends"),
    ]),
    # Pronoms conjoints et négation devant un verbe à h aspiré : pas d'élision.
    "clitique_h_aspire": (DISJONCTION, [
        ("Le loup {} plus dans la forêt.", "ne", "hurle"),
        ("Le chat {} devant le chien.", "se", "hérisse"),
        ("Je {} de rentrer avant le soir.", "me", "hâte"),
        ("Ce souvenir {} encore aujourd'hui.", "le", "hante"),
        ("Le matelot {} au sommet du mât.", "la", "hisse"),
        ("Je {} point ce pauvre garçon.", "ne", "hais"),
    ]),
    # Démonstratif ce devant le verbe être.
    "ce_etre": (ELISION, [
        ("{} la plus belle saison de l'année.", "Ce", "est"),
        ("{} un soir d'hiver, près du feu.", "Ce", "était"),
    ]),
}


def _elider(mot: str) -> str:
    """Forme élidée : la voyelle finale devient une apostrophe droite.

    « le » → « l' », « Je » → « J' », « que » → « qu' », « Lorsque » →
    « Lorsqu' », « jusque » → « jusqu' », « si » → « s' ».
    """
    assert mot[-1] in "aei", mot
    return mot[:-1] + "'"


def generate() -> list:
    """Liste de dicts {"good": str, "bad": str, "phenomene": PHENOMENE, "gabarit": str}.

    L'ordre est fixe (ordre de déclaration des gabarits puis des cadres) :
    aucune part d'aléatoire.
    """
    paires = []
    for gabarit, (regle, cadres) in GABARITS.items():
        for cadre, mot, suivant in cadres:
            elidee = cadre.format(_elider(mot) + suivant)
            pleine = cadre.format(mot + " " + suivant)
            if regle == ELISION:
                bonne, mauvaise = elidee, pleine
            else:
                bonne, mauvaise = pleine, elidee
            paires.append({
                "good": bonne,
                "bad": mauvaise,
                "phenomene": PHENOMENE,
                "gabarit": gabarit,
            })
    return paires


if __name__ == "__main__":
    # Affichage rapide : python evals/paires/elision.py
    _paires = generate()
    for _p in _paires:
        print(f"[{_p['gabarit']}] {_p['good']}  |  *{_p['bad']}")
    _courtes = sum(len(p["good"]) < len(p["bad"]) for p in _paires)
    print(f"{len(_paires)} paires, bonne phrase plus courte : "
          f"{100 * _courtes / len(_paires):.1f} %")
