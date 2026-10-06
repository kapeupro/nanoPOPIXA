"""Paires minimales : accord sujet-verbe en nombre (et en personne).

Chaque paire oppose une phrase grammaticale (« good ») à une phrase
agrammaticale (« bad ») qui ne diffère QUE par la forme conjuguée du verbe
testé : tout le reste est identique caractère pour caractère.

Temps couverts : présent, imparfait, futur simple (à peu près un tiers
chacun ; le temps est noté en commentaire dans les données).

Gabarits (identifiant court dans le champ « gabarit ») :
  - sujet_nominal : sujet nominal simple, sans attracteur
        « Le vent souffle fort sur la lande. » / « *Le vent soufflent ... »
  - attracteur_complement : sujet « N1 de N2 » avec N1 et N2 de nombres
    opposés ; la forme fautive s'accorde avec l'attracteur N2
        « Le chat des voisins dort sur le toit. » / « *... dorment ... »
  - attracteur_relative : sujet « N1 que N2 V2 » ; la relative contient un
    sujet de nombre opposé, la forme fautive s'accorde avec lui
  - relative_qui : verbe d'une relative sujet « N qui V », qui s'accorde
    avec l'antécédent N ; le sujet de la principale, de nombre opposé, sert
    d'attracteur (« Les touristes admirent le pont qui enjambe ... »)
  - pronom_nombre : pronom sujet (je, il, elle, on, ils, elles), la forme
    fautive change le nombre (« *Il marchaient », « *On frappent »)
  - pronom_personne : pronom sujet (je, tu, nous, vous), la forme fautive
    garde le nombre mais change la personne (« *Tu viendra », « *Nous
    aimez ») ; ces erreurs sont souvent plus difficiles (la 3e personne
    fautive est la plus fréquente dans les corpus), d'où un gabarit séparé
    pour ne pas mélanger les deux difficultés dans les rapports
  - sujet_inverse : sujet postposé après un complément circonstanciel ou
    « où » (inversion fréquente chez Hugo) ; plusieurs cadres contiennent un
    nom pluriel placé avant un sujet singulier (« Au fond des bois coulait
    un petit ruisseau. »)
  - clitique_objet : un pronom objet (le, la, les, l') de nombre opposé
    s'intercale entre le sujet et le verbe (attracteur adjacent) ; la forme
    fautive s'accorde avec le clitique

Équilibre : 35 paires à sujet singulier, 34 à sujet pluriel (pour
relative_qui, c'est le nombre de l'antécédent qui compte). Avec un sujet
singulier, la forme fautive (plurielle) est en général plus longue ; avec un
sujet pluriel, c'est l'inverse. La bonne phrase est donc strictement la plus
courte dans environ la moitié des paires (34/69, 49 %). Les seuls écarts à
la règle « singulier = bonne plus courte, pluriel = bonne plus longue »
viennent des erreurs de personne, imposées par le phénomène :
« Tu viendras / *Tu viendra » (sujet singulier, bonne plus longue),
« Je finirai / *Je finiras » et « Nous partirons / *Nous partiront »
(longueurs égales), « Vous chantiez / *Vous chantaient » (sujet pluriel,
bonne plus courte).

Précautions linguistiques :
  - aucun nom collectif ou de quantité en tête du sujet (« la plupart »,
    « une foule de »...), qui autoriseraient les deux accords ;
  - aucune forme « je + -ons » (« je sommes », « j'avons ») : c'est le
    parler paysan des comédies de Molière, pas une faute « sans conteste » ;
  - pas de « Je est » (Rimbaud) ;
  - pas de sujets coordonnés (« X et Y ») : l'accord avec le sujet le plus
    proche était admis dans la langue classique ;
  - verbes choisis pour ne pas être homographes d'un adjectif plausible à la
    même place (pas de « ferme », « calme », « vide »...) ni d'une forme
    d'un autre verbe qui rendrait la phrase fautive localement naturelle
    (pas de « vit », passé simple de « voir ») ;
  - aucun attribut ni adjectif accordé avec le sujet à droite du verbe
    (pas de « est plein », « vivait seul ») : il donnerait un second indice
    d'accord, local, qui rendrait les paires à attracteur trop faciles ;
  - sujets postposés pluriels : aucun des verbes des tours figés qui
    admettent le singulier (« Reste les... », « Vive les... »).

Module autonome : bibliothèque standard uniquement, aucun aléatoire.
"""

PHENOMENE = "accord_sujet_verbe"

# Nombre grammatical du sujet (documentation des données ci-dessous).
SG = "sg"
PL = "pl"


# ---------------------------------------------------------------------------
# Sujet nominal simple : « {sujet} {verbe} {suite}. »
# Entrées : (nombre, sujet, forme correcte, forme fautive, suite)
# ---------------------------------------------------------------------------

_SUJET_NOMINAL = [
    # présent
    (SG, "Le boulanger", "prépare", "préparent", "le pain avant l'aube"),
    (PL, "Les oiseaux", "chantent", "chante", "dans les arbres du parc"),
    (SG, "Ma sœur", "lit", "lisent", "un roman près de la fenêtre"),
    (PL, "Les marins", "réparent", "répare", "leurs filets sur le quai"),
    (SG, "Le vent", "souffle", "soufflent", "fort sur la lande"),
    # imparfait
    (SG, "La vieille femme", "racontait", "racontaient", "des histoires au coin du feu"),
    (PL, "Les paysans", "travaillaient", "travaillait", "aux champs du matin au soir"),
    (SG, "Le roi", "donnait", "donnaient", "de grandes fêtes au château"),
    (PL, "Les chevaux", "attendaient", "attendait", "devant l'auberge depuis une heure"),
    (PL, "Les cloches", "sonnaient", "sonnait", "pour la messe du dimanche"),
    # futur
    (SG, "Le train", "partira", "partiront", "à huit heures précises"),
    (PL, "Les invités", "arriveront", "arrivera", "après le coucher du soleil"),
    (SG, "Le maire", "ouvrira", "ouvriront", "la fête demain matin"),
    (PL, "Les élèves", "passeront", "passera", "leur examen la semaine prochaine"),
    (PL, "Les pêcheurs", "rentreront", "rentrera", "au port avec la marée"),
]


# ---------------------------------------------------------------------------
# Attracteur dans un complément du nom : « {tête} {complément} {verbe} {suite}. »
# La tête et le complément ont des nombres opposés.
# Entrées : (nombre de la tête, tête, complément, correcte, fautive, suite)
# ---------------------------------------------------------------------------

_ATTRACTEUR_COMPLEMENT = [
    # présent
    (SG, "Le chat", "des voisins", "dort", "dorment", "sur le toit"),
    (SG, "Le bruit", "des voitures", "empêche", "empêchent", "le bébé de dormir"),
    (PL, "Les fenêtres", "de la chambre", "donnent", "donne", "sur le jardin"),
    (PL, "Les tours", "du château", "dominent", "domine", "la vallée"),
    # complément invariable après le verbe (pas d'attribut accordé)
    (SG, "Le jardin", "des voisins", "est", "sont", "à vendre"),
    # imparfait (« est » est évité dans les phrases fautives : « la porte est »
    # pourrait se lire « la porte orientale », groupe nominal sans verbe)
    (PL, "Les clés", "de la porte", "étaient", "était", "sous le paillasson"),
    (PL, "Les amis", "de mon frère", "venaient", "venait", "souvent à la maison"),
    (SG, "La voix", "des enfants", "résonnait", "résonnaient", "dans l'église"),
    # futur
    (SG, "Le capitaine", "des soldats", "choisira", "choisiront", "la route"),
    (PL, "Les feuilles", "de l'arbre", "jauniront", "jaunira", "en automne"),
    (SG, "Le prix", "des légumes", "augmentera", "augmenteront", "bientôt"),
    (PL, "Les enfants", "de la voisine", "iront", "ira", "à la plage"),
]


# ---------------------------------------------------------------------------
# Attracteur dans une relative objet :
#   « {tête} que {sujet de la relative} {verbe relatif} {verbe} {suite}. »
# Le sujet de la relative a le nombre opposé à celui de la tête.
# Entrées : (nombre de la tête, tête, relative, correcte, fautive, suite)
# ---------------------------------------------------------------------------

_ATTRACTEUR_RELATIVE = [
    # présent
    (SG, "Le chien", "mes filles promènent", "aboie", "aboient", "très fort"),
    (PL, "Les fleurs", "ma mère arrose", "poussent", "pousse", "très vite"),
    (SG, "Le gâteau", "mes cousins préfèrent", "coûte", "coûtent", "cher"),
    (PL, "Les chemins", "le chasseur connaît", "mènent", "mène", "à la source"),
    # imparfait
    (SG, "Le vieil homme", "les villageois saluaient", "vivait", "vivaient", "près du lac"),
    (PL, "Les tableaux", "le peintre vendait", "valaient", "valait", "une fortune"),
    (PL, "Les chansons", "ma grand-mère chantait", "parlaient", "parlait", "d'amour"),
    # futur
    (SG, "La maison", "mes parents achèteront", "aura", "auront", "un grand jardin"),
    (PL, "Les pommiers", "le jardinier plantera", "donneront", "donnera", "des fruits"),
    (SG, "Le bal", "mes amies organiseront", "durera", "dureront", "toute la nuit"),
]


# ---------------------------------------------------------------------------
# Relative sujet : « {principale} {antécédent} qui {verbe} {suite}. »
# Le verbe s'accorde avec l'antécédent ; le sujet de la principale a le
# nombre opposé (attracteur). Le nombre noté est celui de l'antécédent.
# Entrées : (nombre, principale, antécédent, correcte, fautive, suite)
# ---------------------------------------------------------------------------

_RELATIVE_QUI = [
    # présent
    (SG, "Les touristes admirent", "le pont", "enjambe", "enjambent", "le fleuve"),
    (PL, "Le guide prendra", "les sentiers", "montent", "monte", "vers le col"),
    # imparfait
    (PL, "La jeune fille regardait", "les vagues", "frappaient", "frappait", "la falaise"),
    (SG, "Les femmes écoutaient", "le vieillard", "parlait", "parlaient", "de la guerre"),
    # futur
    (PL, "Ma tante plantera", "des rosiers", "fleuriront", "fleurira", "au printemps"),
    (SG, "Mes oncles achèteront", "le cheval", "gagnera", "gagneront", "la course"),
]


# ---------------------------------------------------------------------------
# Pronoms personnels sujets : « {pronom} {verbe} {suite}. »
# Entrées : (nombre, pronom, correcte, fautive, suite)
# ---------------------------------------------------------------------------

# Erreurs de nombre (« *Je pensent » change aussi la personne).
_PRONOM_NOMBRE = [
    (SG, "Il", "marchait", "marchaient", "lentement le long de la rivière"),
    (PL, "Ils", "cherchaient", "cherchait", "leur chemin dans la forêt"),
    (SG, "Je", "pense", "pensent", "souvent à mon enfance"),
    (SG, "On", "frappe", "frappent", "à la porte"),
    (PL, "Elles", "riaient", "riait", "de bon cœur"),
    (SG, "Elle", "connaît", "connaissent", "le nom de toutes les étoiles"),
    (PL, "Ils", "savent", "sait", "lire et écrire"),
]

# Erreurs de personne seule (même nombre).
_PRONOM_PERSONNE = [
    (PL, "Nous", "partirons", "partiront", "demain à l'aube"),
    (SG, "Tu", "viendras", "viendra", "avec nous au marché"),
    (PL, "Vous", "chantiez", "chantaient", "mieux que nous"),
    (SG, "Je", "finirai", "finiras", "ce travail avant la fin du mois"),
    (PL, "Nous", "aimons", "aimez", "les longues soirées d'hiver"),
]


# ---------------------------------------------------------------------------
# Sujet inversé (postposé) : « {cadre} {verbe} {sujet}. »
# Entrées : (nombre, cadre, correcte, fautive, sujet)
# ---------------------------------------------------------------------------

_SUJET_INVERSE = [
    # présent
    (PL, "Voici la ville où", "grandissent", "grandit", "mes cousins"),
    (SG, "Dans le ciel", "vole", "volent", "un grand oiseau blanc"),
    # imparfait
    (PL, "Dans ce manoir", "habitaient", "habitait", "trois sœurs"),
    (SG, "Au fond des bois", "coulait", "coulaient", "un petit ruisseau"),
    (PL, "Sur la place du village", "dansaient", "dansait", "les jeunes gens"),
    (SG, "Derrière la colline se", "cachait", "cachaient", "un vieux moulin"),
    # futur (attracteur pluriel avant un sujet singulier)
    (SG, "Entre les arbres", "apparaîtra", "apparaîtront", "la lune"),
    (SG, "Sur les remparts", "flottera", "flotteront", "le drapeau du roi"),
]


# ---------------------------------------------------------------------------
# Pronom objet intercalé : « {sujet} {clitique}{verbe} {suite}. »
# Le clitique a le nombre opposé à celui du sujet (attracteur adjacent) ;
# un clitique élidé (« l' ») se colle au verbe.
# Entrées : (nombre, sujet, clitique, correcte, fautive, suite)
# ---------------------------------------------------------------------------

_CLITIQUE_OBJET = [
    (SG, "Le berger", "les", "conduit", "conduisent", "chaque soir à l'étable"),
    (PL, "Mes amis", "le", "respectent", "respecte", "depuis longtemps"),
    (SG, "L'institutrice", "les", "accueillait", "accueillaient", "devant l'école"),
    (PL, "Les gardes", "la", "surveillaient", "surveillait", "jour et nuit"),
    (SG, "Mon père", "les", "appellera", "appelleront", "dimanche prochain"),
    (PL, "Les voisins", "l'", "aideront", "aidera", "à déménager"),
]


def _phrase(*morceaux):
    """Assemble les morceaux avec des espaces simples et ajoute le point final.

    Un morceau qui finit par une apostrophe (élision) se colle au suivant.
    """
    texte = ""
    for morceau in morceaux:
        if texte and not texte.endswith("'"):
            texte += " "
        texte += morceau
    return texte + "."


def _paire(gabarit, avant, correcte, fautive, apres):
    """Construit une paire en ne faisant varier que la forme verbale."""
    return {
        "good": _phrase(*avant, correcte, apres),
        "bad": _phrase(*avant, fautive, apres),
        "phenomene": PHENOMENE,
        "gabarit": gabarit,
    }


def generate() -> list:
    """Liste de dicts {"good": str, "bad": str, "phenomene": PHENOMENE, "gabarit": str}.

    L'ordre est fixe (aucun aléatoire) : les gabarits se suivent dans l'ordre
    de déclaration du module.
    """
    paires = []

    for _nombre, sujet, correcte, fautive, suite in _SUJET_NOMINAL:
        paires.append(_paire("sujet_nominal", (sujet,), correcte, fautive, suite))

    for _nombre, tete, complement, correcte, fautive, suite in _ATTRACTEUR_COMPLEMENT:
        paires.append(_paire("attracteur_complement", (tete, complement),
                             correcte, fautive, suite))

    for _nombre, tete, relative, correcte, fautive, suite in _ATTRACTEUR_RELATIVE:
        paires.append(_paire("attracteur_relative", (tete, "que", relative),
                             correcte, fautive, suite))

    for _nombre, principale, antecedent, correcte, fautive, suite in _RELATIVE_QUI:
        paires.append(_paire("relative_qui", (principale, antecedent, "qui"),
                             correcte, fautive, suite))

    for gabarit, donnees in (("pronom_nombre", _PRONOM_NOMBRE),
                             ("pronom_personne", _PRONOM_PERSONNE)):
        for _nombre, pronom, correcte, fautive, suite in donnees:
            paires.append(_paire(gabarit, (pronom,), correcte, fautive, suite))

    for _nombre, cadre, correcte, fautive, sujet in _SUJET_INVERSE:
        paires.append(_paire("sujet_inverse", (cadre,), correcte, fautive, sujet))

    for _nombre, sujet, clitique, correcte, fautive, suite in _CLITIQUE_OBJET:
        paires.append(_paire("clitique_objet", (sujet, clitique),
                             correcte, fautive, suite))

    return paires


if __name__ == "__main__":
    for p in generate():
        print("[{}]\n  + {}\n  - {}".format(p["gabarit"], p["good"], p["bad"]))
