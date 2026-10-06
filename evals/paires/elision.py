"""Paires minimales : élision et h aspiré.

Phénomène testé (à la BLiMP) : devant une voyelle ou un h muet, les mots
grammaticaux brefs le, la, de, que (lorsque, jusque), je, me, te, se, ne, ce
perdent leur voyelle finale, remplacée par une apostrophe ; « si » ne s'élide
que devant « il » et « ils ». L'élision est en revanche interdite devant un
h aspiré, après « si » devant elle / on / un, et pour « le », « la » placés
après un impératif. La phrase correcte et la phrase fautive ne diffèrent que
par cette élision.

Quatre familles de paires :

  V  élision devant voyelle (la bonne phrase est la forme élidée) :
         « l'arbre » / « *le arbre », « d'or » / « *de or »,
         « qu'elle » / « *que elle », « n'a qu'à » / « *n'a que à »,
         « jusqu'au » / « *jusque au », « s'il » / « *si il »,
         « n'y » / « *ne y », « c'est » / « *ce est », « c'en » / « *ce en » ;
  M  élision devant h muet (la bonne phrase est la forme élidée) :
         « l'homme », « l'heure », « d'habitude », « d'huile », « j'habite »,
         « s'habille », « n'hésite », « l'honore »... ;
  A  h aspiré, pas d'élision (la bonne phrase est la forme pleine) :
         « le héros » / « *l'héros », « de honte » / « *d'honte »,
         « je hais » / « *j'hais », « se hisse » / « *s'hisse »... ;
  S  autre voyelle, pas d'élision (la bonne phrase est la forme pleine) :
         « si elle » / « *s'elle », « si on » / « *s'on », « si un » /
         « *s'un » ; « Laisse-le entrer » / « *Laisse-l'entrer » (le pronom
         placé après l'impératif est tonique et ne s'élide jamais devant une
         voyelle, sauf devant en et y, cas exclus ici).

Ce que mesure le banc : une règle purement graphique (« élider devant une
voyelle écrite, jamais devant h ») ne réussit que les familles V et A ; elle
échoue sur M et S. Les familles sont donc équilibrées pour que cette règle
reste près du hasard : 20 V, 17 M, 17 A, 14 S (68 paires), soit 37 paires
sur 68 (54,4 %) gagnées par l'heuristique graphique. Autant de h muets que
de h aspirés, répartis dans les mêmes gabarits (article, de, clitiques) : le
modèle doit connaître le mot, pas seulement sa première lettre. Plusieurs
cadres opposent un h muet à un h aspiré du même champ lexical (« l'héroïne »
/ « le héros », « n'hésite » / « ne hurle », « plein d'espoir » / « plein de
haine », « qu'elle » / « si elle »).

Équilibre des longueurs : l'élision raccourcit toujours la phrase d'un
caractère (voyelle + espace remplacées par l'apostrophe). La longueur est
donc entièrement liée à la règle, ce qu'impose le phénomène : la bonne
phrase est la plus courte dans les familles V et M (37 paires, 54,4 %) et la
plus longue dans les familles A et S (31 paires, 45,6 %). Un modèle qui
préfère les phrases courtes ne gagne rien à cette préférence. Aucune paire
n'a deux phrases de même longueur.

Construction : chaque entrée est (cadre, mot élidable, mot suivant). Le cadre
contient une case « {} » qui reçoit soit « mot suivant » (forme pleine), soit
la forme élidée (« le » → « l' », « lorsque » → « lorsqu' »...) collée au mot
suivant. Le reste de la phrase est donc identique caractère pour caractère
(différence minimale garantie par construction). La règle du gabarit
(ELISION ou DISJONCTION) dit laquelle des deux formes est la bonne. Un
gabarit = un mot grammatical (ou un groupe de mots de même nature) × une
famille, pour que les scores par gabarit restent lisibles.

Précautions linguistiques :
  - seuls des h aspirés et des h muets stables dans l'usage écrit, classique
    comme moderne, sont retenus. Sont exclus les mots dont l'usage a hésité :
    hérisser et hérisson (« s'hérisser » chez des écrivains, selon Littré et
    Grevisse), hyène, handicap, hiatus, ouate, ouistiti, haricot, ainsi que
    « onze » et « huit », dont l'élision (« l'onzième ») se lit encore chez
    les classiques ; pas de nom propre (« de Hugo » / « d'Hugo » hésite) ;
  - pas de « que » devant le numéral « un » : on peut ne pas élider pour
    insister sur le nombre (« plus de un », « de un à dix ») ; le « que »
    restrictif est testé devant « à » (« n'a qu'à »), sans numéral ;
  - pas de « presque » ni de « quelque » (« presqu'entier »,
    « quelqu'autre » se rencontrent dans les textes anciens) ;
  - « s'elle » et « s'on » n'appartiennent qu'au moyen français : chez
    Molière comme aujourd'hui, seul « si » + elle / on / un est admis ;
  - après l'impératif, aucun pronom n'est suivi de « en » ou « y » (où
    l'élision redevient de règle : « Mets-l'y ») ;
  - aucun article masculin après « à » ou « de » dans les cadres : la forme
    fautive « *à le héros » cumulerait deux fautes (contraction et élision),
    et « au héros » / « *à l'héros » ne serait plus une différence minimale ;
  - aucune question (toutes les phrases finissent par un point) ; les autres
    apostrophes du cadre (« aujourd'hui », « l'année », « d'hiver ») sont
    identiques dans les deux phrases.

Module autonome, bibliothèque standard uniquement, sortie déterministe.
"""

PHENOMENE = "elision"

# Règle d'un gabarit : quelle forme est la bonne.
ELISION = "elision"          # élision obligatoire : la forme élidée est correcte
DISJONCTION = "disjonction"  # élision interdite : la forme pleine est correcte

# Gabarit -> (règle, liste de (cadre, mot élidable, mot suivant)).
# Le cadre contient exactement une case « {} ».
GABARITS = {
    # --- Article défini le / la -------------------------------------------
    # Devant une voyelle (famille V).
    "article_voyelle": (ELISION, [
        ("Le paysan plante {} au bout du champ.", "le", "arbre"),
        ("{} de la rivière est très froide.", "La", "eau"),
        ("{} joue seul dans la cour.", "Le", "enfant"),
        ("{} gronde au-dessus de la ville.", "Le", "orage"),
    ]),
    # Devant un h muet (famille M).
    "article_h_muet": (ELISION, [
        ("{} marche lentement sur la route.", "Le", "homme"),
        ("Les vaches mangent {} du pré.", "la", "herbe"),
        ("{} arrive avec la neige et le froid.", "Le", "hiver"),
        ("Le vieux marin raconte {} de son naufrage.", "la", "histoire"),
        ("{} du roman meurt à la fin.", "La", "héroïne"),
        ("{} du départ approche enfin.", "La", "heure"),
        ("Le voyageur cherche {} de la gare.", "le", "hôtel"),
    ]),
    # Devant un h aspiré : pas d'élision (famille A).
    "article_h_aspire": (DISJONCTION, [
        ("{} revient enfin dans son village.", "Le", "héros"),
        ("Le bûcheron saisit {} sans un mot.", "la", "hache"),
        ("On entend {} au fond du bois.", "le", "hibou"),
        ("Le jardinier taille {} devant la maison.", "la", "haie"),
        ("{} fait bien les choses.", "Le", "hasard"),
        ("{} traverse le marais sans bruit.", "Le", "héron"),
        ("Mon grand-père a mal à {}.", "la", "hanche"),
    ]),

    # --- Préposition de -----------------------------------------------------
    # Devant une voyelle (famille V).
    "de_voyelle": (ELISION, [
        ("Elle a beaucoup {} au village.", "de", "amis"),
        ("Le jeune soldat est plein {}.", "de", "espoir"),
        ("Le roi porte une couronne {}.", "de", "or"),
    ]),
    # Devant un h muet (famille M).
    "de_h_muet": (ELISION, [
        ("{}, il se lève avant le jour.", "De", "habitude"),
        ("Elle achète une bouteille {}.", "de", "huile"),
        ("Il a donné sa parole {}.", "de", "honneur"),
        ("Elle pousse un cri {}.", "de", "horreur"),
    ]),
    # Devant un h aspiré : pas d'élision (famille A).
    "de_h_aspire": (DISJONCTION, [
        ("Il rougit {} devant son père.", "de", "honte"),
        ("Son cœur est plein {}.", "de", "haine"),
        ("Le mur mesure deux mètres {}.", "de", "hauteur"),
        ("Le mendiant est vêtu {}.", "de", "haillons"),
    ]),

    # --- Clitiques préverbaux : je, me, se, le et la négation ne ------------
    # Devant une voyelle (famille V).
    "clitique_voyelle": (ELISION, [
        ("{} la musique de mon pays.", "Je", "aime"),
        ("Ma mère {} depuis le jardin.", "me", "appelle"),
        ("Il {} a plus de pain à la maison.", "ne", "y"),
    ]),
    # Devant un verbe à h muet (famille M).
    "clitique_h_muet": (ELISION, [
        ("{} près de la gare depuis un an.", "Je", "habite"),
        ("{} de la maison de mon oncle.", "Je", "hérite"),
        ("Elle {} vite le matin.", "se", "habille"),
        ("Le chien {} à son nouveau maître.", "se", "habitue"),
        ("Le capitaine {} jamais devant le danger.", "ne", "hésite"),
        ("Tout le village {} comme un saint.", "le", "honore"),
    ]),
    # Devant un verbe à h aspiré : pas d'élision (famille A).
    "clitique_h_aspire": (DISJONCTION, [
        ("{} le mensonge et la flatterie.", "Je", "hais"),
        ("{} les épaules sans répondre.", "Je", "hausse"),
        ("Le loup {} plus dans la forêt.", "ne", "hurle"),
        ("Je {} de rentrer avant le soir.", "me", "hâte"),
        ("Ce souvenir {} encore aujourd'hui.", "le", "hante"),
        ("Le chat {} sur le mur du jardin.", "se", "hisse"),
    ]),

    # --- Autres mots élidables devant voyelle (famille V) -------------------
    # Que (conjonction ou restrictif ne... que) et ses composés lorsque, jusque.
    "que_et_composes": (ELISION, [
        ("Il faut {} parte avant la nuit.", "que", "elle"),
        ("Le pauvre berger n'a {} attendre le printemps.", "que", "à"),
        ("{} entra, tout le monde se tut.", "Lorsque", "il"),
        ("Nous avons dansé {} matin.", "jusque", "au"),
    ]),
    # Si s'élide devant il / ils.
    "si_il": (ELISION, [
        ("{} pleut demain, nous resterons ici.", "Si", "il"),
        ("Je ne sais pas {} dort encore.", "si", "il"),
        ("{} arrivent tôt, nous dînerons ensemble.", "Si", "ils"),
    ]),
    # Pronom démonstratif ce devant le verbe être et devant en.
    "ce_elide": (ELISION, [
        ("{} la plus belle saison de l'année.", "Ce", "est"),
        ("{} un soir d'hiver, près du feu.", "Ce", "était"),
        ("{} est trop pour moi.", "Ce", "en"),
    ]),

    # --- Voyelle sans élision (famille S) -----------------------------------
    # Si ne s'élide pas devant elle(s), on, un(e).
    "si_sans_elision": (DISJONCTION, [
        ("{} vient, nous partirons à midi.", "Si", "elle"),
        ("Je me demande {} viendra ce soir.", "si", "elle"),
        ("{} frappe, ouvre la porte.", "Si", "on"),
        ("{} enfant pleure, il faut le consoler.", "Si", "un"),
        ("{} lettre arrive, préviens-moi.", "Si", "une"),
        ("Demande-lui {} peut entrer.", "si", "on"),
        ("{} chantent, le public applaudira.", "Si", "elles"),
    ]),
    # Le / la placés après un impératif ne s'élident pas devant une voyelle.
    "imperatif_sans_elision": (DISJONCTION, [
        ("Laisse-{} dans la maison.", "le", "entrer"),
        ("Mets-{} chaud dans la cuisine.", "le", "au"),
        ("Accompagne-{} la gare demain matin.", "la", "à"),
        ("Garde-{} toi jusqu'au soir.", "la", "avec"),
        ("Invite-{} dîner ce soir.", "le", "à"),
        ("Fais-{} près de la fenêtre.", "la", "asseoir"),
        ("Pose-{}, sur la table.", "le", "ici"),
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
