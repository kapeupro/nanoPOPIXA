"""Paires minimales : accord sujet-verbe en nombre (et en personne).

Chaque paire oppose une phrase grammaticale (« good ») à une phrase
agrammaticale (« bad ») qui ne diffère QUE par la forme conjuguée du verbe
testé : tout le reste est identique caractère pour caractère.

Temps couverts : présent, imparfait, futur simple.

Gabarits (identifiant court dans le champ « gabarit ») :
  - sn_present / sn_imparfait / sn_futur : sujet nominal simple
        « Le vent souffle fort sur la lande. » / « *Le vent soufflent ... »
  - attracteur_complement : sujet « N1 de N2 » avec N1 et N2 de nombres
    opposés ; la forme fautive s'accorde avec l'attracteur N2
        « Le chat des voisins dort sur le toit. » / « *... dorment ... »
  - attracteur_relative : sujet « N1 que N2 V2 » ; la relative contient un
    sujet de nombre opposé, la forme fautive s'accorde avec lui
  - pronom_personnel : je, tu, il, elle, on, nous, vous, ils, elles
    (erreurs de nombre et de personne)
  - sujet_inverse : sujet postposé après un complément circonstanciel ou
    « où » (inversion fréquente chez Hugo)
  - clitique_objet : un pronom objet (le, la, les, nous) de nombre opposé
    s'intercale entre le sujet et le verbe

Équilibre : 34 paires à sujet singulier, 34 à sujet pluriel. Avec un sujet
singulier, la forme fautive (plurielle) est en général plus longue ; avec un
sujet pluriel, c'est l'inverse. La bonne phrase est donc strictement la plus
courte dans environ la moitié des paires (33/68). Les seuls écarts à la règle
« singulier = plus court » viennent des erreurs de personne, imposées par le
phénomène : « Tu viendras / *Tu viendra » (bonne plus longue),
« Je finirai / *Je finiras » et « Nous partirons / *Nous partiront »
(longueurs égales), « Vous chantiez / *Vous chantaient » (bonne plus courte),
« Nous aimons / *Nous aimez » (bonne plus longue).

Précautions linguistiques :
  - aucun nom collectif ou de quantité en tête du sujet (« la plupart »,
    « une foule de »...), qui autoriseraient les deux accords ;
  - aucune forme « je + -ons » (« je sommes », « j'avons ») : c'est le
    parler paysan des comédies de Molière, pas une faute « sans conteste » ;
  - pas de « Je est » (Rimbaud) ;
  - verbes choisis pour ne pas être homographes d'un adjectif plausible à la
    même place (pas de « ferme », « calme », « vide »...).

Module autonome : bibliothèque standard uniquement, aucun aléatoire.
"""

PHENOMENE = "accord_sujet_verbe"

# Nombre grammatical du sujet (documentation des données ci-dessous).
SG = "sg"
PL = "pl"


# ---------------------------------------------------------------------------
# Gabarits sujet nominal simple : « {sujet} {verbe} {suite}. »
# Entrées : (nombre, sujet, forme correcte, forme fautive, suite)
# ---------------------------------------------------------------------------

_SN_PRESENT = [
    (SG, "Le boulanger", "prépare", "préparent", "le pain avant l'aube"),
    (PL, "Les oiseaux", "chantent", "chante", "dans les arbres du parc"),
    (SG, "Ma sœur", "lit", "lisent", "un roman près de la fenêtre"),
    (PL, "Les marins", "réparent", "répare", "leurs filets sur le quai"),
    (SG, "Le vent", "souffle", "soufflent", "fort sur la lande"),
    (PL, "Mes parents", "habitent", "habite", "une petite maison à la campagne"),
    (SG, "Le médecin", "soigne", "soignent", "les malades de la ville"),
    (PL, "Les étudiants", "écoutent", "écoute", "le professeur en silence"),
]

_SN_IMPARFAIT = [
    (SG, "La vieille femme", "racontait", "racontaient", "des histoires au coin du feu"),
    (PL, "Les paysans", "travaillaient", "travaillait", "aux champs du matin au soir"),
    (SG, "Le roi", "donnait", "donnaient", "de grandes fêtes au château"),
    (PL, "Les chevaux", "attendaient", "attendait", "devant la porte de l'auberge"),
    (SG, "Mon grand-père", "fumait", "fumaient", "la pipe après le dîner"),
    (PL, "Les cloches", "sonnaient", "sonnait", "pour la messe du dimanche"),
    (SG, "Le petit garçon", "pleurait", "pleuraient", "dans les bras de sa mère"),
    (PL, "Les étoiles", "brillaient", "brillait", "au-dessus de la mer"),
]

_SN_FUTUR = [
    (SG, "Le train", "partira", "partiront", "à huit heures précises"),
    (PL, "Les invités", "arriveront", "arrivera", "après le coucher du soleil"),
    (SG, "Le maire", "ouvrira", "ouvriront", "la fête demain matin"),
    (PL, "Les élèves", "passeront", "passera", "leur examen la semaine prochaine"),
    (SG, "La neige", "couvrira", "couvriront", "les montagnes tout l'hiver"),
    (PL, "Les pommes", "tomberont", "tombera", "bientôt de l'arbre"),
    (SG, "Votre frère", "viendra", "viendront", "nous voir cet été"),
    (PL, "Les pêcheurs", "rentreront", "rentrera", "au port avant la tempête"),
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
    (SG, "Le jardin", "des voisins", "est", "sont", "plein de roses"),
    # imparfait (« est » est évité dans les phrases fautives : « la porte est »
    # pourrait se lire « la porte orientale », groupe nominal sans verbe)
    (PL, "Les clés", "de la porte", "étaient", "était", "sous le paillasson"),
    (PL, "Les amis", "de mon frère", "venaient", "venait", "souvent à la maison"),
    (SG, "La voix", "des enfants", "résonnait", "résonnaient", "dans l'église"),
    # futur
    (SG, "Le capitaine", "des soldats", "choisira", "choisiront", "la route"),
    (PL, "Les feuilles", "de l'arbre", "jauniront", "jaunira", "en automne"),
    (SG, "Le prix", "des légumes", "augmentera", "augmenteront", "cet hiver"),
    (PL, "Les enfants", "de la voisine", "iront", "ira", "à la plage"),
]


# ---------------------------------------------------------------------------
# Attracteur dans une relative objet :
#   « {tête} que {sujet de la relative} {verbe relatif} {verbe} {suite}. »
# Le sujet de la relative a le nombre opposé à celui de la tête.
# Entrées : (nombre de la tête, tête, relative, correcte, fautive, suite)
# ---------------------------------------------------------------------------

_ATTRACTEUR_RELATIVE = [
    (SG, "Le chien", "les enfants promènent", "aboie", "aboient", "très fort"),
    (PL, "Les fleurs", "ma mère arrose", "poussent", "pousse", "très vite"),
    (SG, "Le gâteau", "mes cousins préfèrent", "coûte", "coûtent", "cher"),
    (PL, "Les chemins", "le berger connaît", "mènent", "mène", "à la source"),
    (SG, "Le vieil homme", "les villageois saluaient", "vivait", "vivaient", "seul"),
    (PL, "Les tableaux", "le peintre vendait", "valaient", "valait", "une fortune"),
    (SG, "La maison", "mes parents achèteront", "aura", "auront", "un grand jardin"),
    (PL, "Les chansons", "ma grand-mère chantait", "parlaient", "parlait", "d'amour"),
]


# ---------------------------------------------------------------------------
# Pronoms personnels sujets : « {pronom} {verbe} {suite}. »
# Erreurs de nombre (« *Il marchaient ») et de personne (« *Tu viendra »).
# Entrées : (nombre, pronom, correcte, fautive, suite)
# ---------------------------------------------------------------------------

_PRONOM_PERSONNEL = [
    (SG, "Il", "marchait", "marchaient", "seul le long de la rivière"),
    (PL, "Ils", "cherchaient", "cherchait", "leur chemin dans la forêt"),
    (SG, "Je", "pense", "pensent", "souvent à mon enfance"),
    (PL, "Nous", "partirons", "partiront", "demain à l'aube"),
    (SG, "Tu", "viendras", "viendra", "avec nous au marché"),
    (PL, "Vous", "chantiez", "chantaient", "mieux que nous"),
    (SG, "On", "frappe", "frappent", "à la porte"),
    (PL, "Elles", "riaient", "riait", "de bon cœur"),
    (SG, "Elle", "connaît", "connaissent", "tous les chemins de la forêt"),
    (PL, "Ils", "savent", "sait", "lire et écrire"),
    (SG, "Je", "finirai", "finiras", "ce travail avant ce soir"),
    (PL, "Nous", "aimons", "aimez", "les longues soirées d'hiver"),
]


# ---------------------------------------------------------------------------
# Sujet inversé (postposé) : « {cadre} {verbe} {sujet}. »
# Entrées : (nombre, cadre, correcte, fautive, sujet)
# ---------------------------------------------------------------------------

_SUJET_INVERSE = [
    (PL, "Dans la vieille maison", "habitaient", "habitait", "trois sœurs"),
    (SG, "Au fond du jardin", "coulait", "coulaient", "un petit ruisseau"),
    (PL, "Sur la place du village", "dansaient", "dansait", "les jeunes gens"),
    (SG, "Derrière la colline se", "cachait", "cachaient", "un vieux moulin"),
    (PL, "Voici la ville où", "vivent", "vit", "mes cousins"),
    (SG, "Dans le ciel", "vole", "volent", "un grand oiseau blanc"),
]


# ---------------------------------------------------------------------------
# Pronom objet intercalé : « {sujet} {clitique} {verbe} {suite}. »
# Le clitique a le nombre opposé à celui du sujet (attracteur adjacent).
# Entrées : (nombre, sujet, clitique, correcte, fautive, suite)
# ---------------------------------------------------------------------------

_CLITIQUE_OBJET = [
    (SG, "Le berger", "les", "conduit", "conduisent", "chaque soir à l'étable"),
    (PL, "Mes amis", "le", "respectent", "respecte", "depuis longtemps"),
    (SG, "L'institutrice", "les", "accueillait", "accueillaient", "devant l'école"),
    (PL, "Les gardes", "la", "surveillaient", "surveillait", "jour et nuit"),
    (SG, "Mon père", "les", "appellera", "appelleront", "dimanche prochain"),
    (PL, "Les voisins", "nous", "aideront", "aidera", "à déménager"),
]


def _phrase(*morceaux):
    """Assemble les morceaux avec des espaces simples et ajoute le point final."""
    return " ".join(morceaux) + "."


def _paire(gabarit, avant, correcte, fautive, apres):
    """Construit une paire en ne faisant varier que la forme verbale."""
    return {
        "good": _phrase(avant, correcte, apres),
        "bad": _phrase(avant, fautive, apres),
        "phenomene": PHENOMENE,
        "gabarit": gabarit,
    }


def generate() -> list:
    """Liste de dicts {"good": str, "bad": str, "phenomene": PHENOMENE, "gabarit": str}.

    L'ordre est fixe (aucun aléatoire) : les gabarits se suivent dans l'ordre
    de déclaration du module.
    """
    paires = []

    for gabarit, donnees in (
        ("sn_present", _SN_PRESENT),
        ("sn_imparfait", _SN_IMPARFAIT),
        ("sn_futur", _SN_FUTUR),
    ):
        for _nombre, sujet, correcte, fautive, suite in donnees:
            paires.append(_paire(gabarit, sujet, correcte, fautive, suite))

    for _nombre, tete, complement, correcte, fautive, suite in _ATTRACTEUR_COMPLEMENT:
        paires.append(_paire("attracteur_complement", tete + " " + complement,
                             correcte, fautive, suite))

    for _nombre, tete, relative, correcte, fautive, suite in _ATTRACTEUR_RELATIVE:
        paires.append(_paire("attracteur_relative", tete + " que " + relative,
                             correcte, fautive, suite))

    for _nombre, pronom, correcte, fautive, suite in _PRONOM_PERSONNEL:
        paires.append(_paire("pronom_personnel", pronom, correcte, fautive, suite))

    for _nombre, cadre, correcte, fautive, sujet in _SUJET_INVERSE:
        paires.append(_paire("sujet_inverse", cadre, correcte, fautive, sujet))

    for _nombre, sujet, clitique, correcte, fautive, suite in _CLITIQUE_OBJET:
        paires.append(_paire("clitique_objet", sujet + " " + clitique,
                             correcte, fautive, suite))

    return paires


if __name__ == "__main__":
    for p in generate():
        print("[{}]\n  + {}\n  - {}".format(p["gabarit"], p["good"], p["bad"]))
