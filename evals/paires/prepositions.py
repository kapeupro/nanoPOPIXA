"""Paires minimales : prépositions, articles contractés et noms de lieux.

Chaque paire oppose une phrase grammaticale (« good ») à une phrase
agrammaticale (« bad ») qui ne diffère QUE par la forme testée : la
préposition, l'article contracté (au, aux, du, des) ou sa forme élidée
(à l', de l', d'). Chaque gabarit est un cadre de phrase avec une case « {} »
où l'on insère la forme correcte ou la forme fautive : tout le reste est
identique caractère pour caractère (différence minimale garantie par
construction). Quand la forme testée est élidée, la case est collée au mot
suivant et la forme non élidée porte son espace (« à l' » / « au »).

Gabarits (identifiant court dans le champ « gabarit ») :
  - contraction_a : à + le -> au, à + les -> aux devant un nom ; devant un
    nom MASCULIN à initiale vocalique, l'article s'élide et ne se contracte
    pas (« à l' », jamais « au »)
        « Ma mère va au marché le samedi. » / « *... à le marché ... »
        « Il a attaché son cheval à l'arbre. » / « *... au arbre. »
  - contraction_de : de + le -> du, de + les -> des (complément du nom) ;
    « de l' » devant un nom masculin à initiale vocalique
        « La maison des voisins a un grand jardin. » / « *... de les voisins ... »
        « La porte de l'atelier reste ouverte. » / « *La porte du atelier ... »
  - partitif : article partitif masculin « du », et « de l' » devant un nom
    masculin à initiale vocalique
        « Le boulanger vend du pain frais. » / « *... de le pain frais. »
        « Le cuisinier ajoute de l'ail ... » / « *... du ail ... »
  - pronom_infinitif : « le » / « les » PRONOMS compléments d'un infinitif
    ne se contractent jamais avec « à » ou « de » (seul l'article se
    contracte) ; ici « à le », « à les », « de le », « de les » sont la
    bonne forme et « au », « aux », « du », « des » la forme fautive
        « Il a commencé à le lire hier soir. » / « *... au lire ... »
        « Nous avons décidé de les inviter. » / « *... des inviter. »
  - ville : « à » (jamais « en ») devant un nom de ville sans article
        « Mon oncle habite à Paris ... » / « *... en Paris ... »
  - pays_feminin : « en » devant un pays féminin
        « Ma tante vit en France ... » / « *... à France ... »
        « ... un long voyage en Italie. » / « *... au Italie. »
  - pays_masculin : « au » devant un pays masculin commençant par une consonne
        « Ma cousine est partie vivre au Japon. » / « *... en Japon. »
  - pays_pluriel : « aux » devant un nom de lieu pluriel (pays, archipels,
    et « les Indes » des textes classiques)
        « Ma sœur fait ses études aux États-Unis. » / « *... en États-Unis. »
  - provenance : « de » (d') devant un pays féminin, « du » / « des » devant
    un pays masculin ou pluriel
        « Mon père revient du Japon demain. » / « *... de Japon demain. »

Indices de surface neutralisés :
  - les suites « à le », « à les », « de le », « de les » figurent dans
    15 mauvaises phrases (article non contracté devant un nom) mais aussi
    dans 12 bonnes phrases (gabarit pronom_infinitif) ; symétriquement,
    « au », « aux », « du », « des » devant un infinitif rendent fautives ces
    12 mauvaises phrases. Un modèle qui pénalise simplement le bigramme
    « à le » ou « de les » ne gagne donc pas ces paires sans connaître la
    règle (contraction de l'article, jamais du pronom) ;
  - « au » ou « du » suivi d'une voyelle n'apparaît que dans des mauvaises
    phrases (9 paires) : ce n'est pas un indice parasite mais la règle
    elle-même (devant voyelle, l'article s'élide au lieu de se contracter ;
    les seules exceptions, h aspiré et « onze », relèvent du module elision
    et sont exclues ici).

Exclusions volontaires (pour que la phrase fautive soit « sans conteste »
agrammaticale, et non simplement ancienne, régionale, administrative ou
défendable) :
  - aucune « sur-contraction » devant un nom féminin (« *au gare »,
    « *du farine ») : la faute y porterait sur le genre lexical du nom,
    phénomène déjà couvert par accord_nominal (gabarit det_defini) ; les
    paires « à l' » / « *au » n'emploient que des noms MASCULINS à initiale
    vocalique (arbre, internat, atelier, été, ail, alcool, argent), pour que
    la seule faute soit la contraction au lieu de l'élision ;
  - pas de nom à « h » (muet ou aspiré : « l'huile », « du hibou »), qui
    relève du phénomène elision, ni de négation (« pas de pain »), qui
    change l'article partitif ;
  - pronom_infinitif : aucun infinitif substantivé (lever, coucher, dîner,
    goûter, savoir, pouvoir, devoir, rire, sourire, toucher, souvenir,
    vivre...), où « au », « du » ou « des » + nom serait correct ;
  - pas de ville dont le nom contient l'article (Le Havre, Le Caire) :
    « *à Le Havre » est condamné par la norme mais courant à l'écrit
    administratif (« né à Le Havre ») ;
  - pas de ville à initiale vocalique (« en Avignon », « en Arles » sont
    admis), ni de ville homonyme d'une région, d'un département ou d'un
    pays (Vienne, Québec, Luxembourg, Panama...), ni de pays dont le nom
    désignait aussi une ville chez les classiques (« à Maroc » =
    Marrakech, « à Mexique » = Mexico) ;
  - jamais « au » ou « du » fautif devant un nom de ville ou devant
    « France » : « au Paris des années folles », « du Paris d'autrefois »
    sont corrects avec un complément, et « le Paris », « le France »
    désignent aussi un café, un navire (le paquebot France) ou un club ;
  - « France » est aussi un prénom (« Il a écrit à France ») : la case est
    toujours un complément circonstanciel de lieu (habiter, vivre, partir,
    se trouver, naître...), jamais après un verbe à complément
    d'attribution (« écrire à », « parler à ») ni après « voir en X » ;
  - pas de « en » fautif devant Portugal, Danemark ou Canada (« en
    Portugal », « en Canada » se lisent chez les classiques) ; « en »
    fautif devant un pays masculin n'est employé que pour « Japon »,
    exemple imposé par la consigne : « en Japon » se rencontre dans
    quelques textes anciens, mais il est nettement agrammatical en français
    moderne ;
  - pas d'île sans article (« à Cuba », « à Malte », « à Chypre » sont
    corrects).

Doubles fautes assumées : « *au Italie », « *du Espagne », « *du Italie »
(dans la lignée de « *au Italie » donné dans la consigne) cumulent genre et
absence d'élision, mais la règle « en / de devant un nom de pays féminin ou à
initiale vocalique » est une seule et même règle. Les autres paires de pays
féminins emploient des noms à initiale consonantique.

Équilibre des longueurs (statistiques sur les 65 paires) :
  - en caractères : bonne phrase strictement plus courte dans 27 paires
    (41,5 %), plus longue dans 27, de même longueur dans 11 ;
  - en tokens gpt2 (le tokenizer du projet ; le score est une somme de
    log-probabilités par token) : bonne phrase plus courte dans 17 paires,
    plus longue dans 22, de même longueur dans 26.
La contraction raccourcit la forme (« au » / « à le ») : les gabarits
contraction_a, contraction_de et partitif mêlent donc des paires où la
contraction est la bonne forme (bonne phrase plus courte) et des paires
d'élision (« de l'ail » / « *du ail »), où la bonne phrase est plus longue en
tokens (et en caractères sauf pour « à l' » / « au ») ; pronom_infinitif
(bonne phrase toujours plus longue) compense les contractions devant un nom.
Orientations imposées par le phénomène :
  - « ville » : bonne phrase toujours plus courte d'un caractère (« à »
    contre « en ») mais de même longueur en tokens ;
  - « pays_feminin » : bonne phrase jamais plus courte en caractères
    (« en » contre « à » ou « au ») et de même longueur en tokens ;
ces deux gabarits se compensent en caractères (8 paires chacun).
« pays_masculin », « pays_pluriel » et « provenance » mêlent les cas en
variant la forme fautive (« en », « à », « à le », « de »...).

Module autonome, bibliothèque standard uniquement, sortie déterministe.
"""

PHENOMENE = "prepositions"

# Gabarit -> liste de (cadre, forme correcte, forme fautive).
# Le cadre contient exactement une case « {} ». Pour les formes élidées, la
# case est collée au mot suivant : la forme non élidée porte alors son espace.
GABARITS = {
    # Préposition « à » + article défini : contraction obligatoire devant
    # « le » / « les » ; élision (et pas de contraction) devant un nom
    # masculin à initiale vocalique.
    "contraction_a": [
        ("Ma mère va {} marché le samedi.", "au", "à le"),
        ("Le maître parle {} enfants avec douceur.", "aux", "à les"),
        ("Nous pensons souvent {} voyage de l'an dernier.", "au", "à le"),
        ("Le vieillard donne du pain {} pauvres.", "aux", "à les"),
        ("Il a attaché son cheval {}arbre.", "à l'", "au "),
        ("Mon frère entre {}internat en septembre.", "à l'", "au "),
    ],
    # Préposition « de » + article défini (complément du nom).
    "contraction_de": [
        ("La maison {} voisins a un grand jardin.", "des", "de les"),
        ("Le bruit {} moulin réveille le village.", "du", "de le"),
        ("Le fils {} roi monte un cheval blanc.", "du", "de le"),
        ("On entend le chant {} oiseaux au printemps.", "des", "de les"),
        ("La porte {}atelier reste ouverte.", "de l'", "du "),
        ("La chaleur {}été entre dans la chambre.", "de l'", "du "),
    ],
    # Article partitif masculin : « du », ou « de l' » devant une voyelle.
    "partitif": [
        ("Le boulanger vend {} pain frais.", "du", "de le"),
        ("Mon grand-père boit {} vin au dîner.", "du", "de le"),
        ("Elle met {} sucre dans son café.", "du", "de le"),
        ("Le cuisinier ajoute {}ail dans la sauce.", "de l'", "du "),
        ("L'infirmière verse {}alcool sur la plaie.", "de l'", "du "),
        ("Mon cousin a gagné {}argent cet automne.", "de l'", "du "),
    ],
    # Pronom complément « le » / « les » devant un infinitif : jamais de
    # contraction avec « à » ou « de ».
    "pronom_infinitif": [
        ("Il a commencé {} lire hier soir.", "à le", "au"),
        ("Elle a réussi {} convaincre.", "à le", "au"),
        ("Nous avons décidé {} inviter.", "de les", "des"),
        ("Le vieux soldat refuse {} suivre.", "de le", "du"),
        ("Il faut penser {} prévenir.", "à les", "aux"),
        ("Le berger essaie {} rassembler.", "de les", "des"),
        ("Le médecin a promis {} soigner.", "de le", "du"),
        ("Les parents cherchent {} protéger.", "à les", "aux"),
        ("Le professeur continue {} encourager.", "à les", "aux"),
        ("Il a oublié {} remercier.", "de les", "des"),
        ("La reine hésite {} punir.", "à le", "au"),
        ("Mon grand-père m'a appris {} reconnaître.", "à les", "aux"),
    ],
    # Nom de ville sans article : « à » (jamais « en »).
    "ville": [
        ("Mon oncle habite {} Paris depuis longtemps.", "à", "en"),
        ("Nous passerons l'hiver {} Rome.", "à", "en"),
        ("Elle a étudié la peinture {} Venise.", "à", "en"),
        ("Le navire arrive {} Marseille demain matin.", "à", "en"),
        ("Mon cousin travaille {} Londres.", "à", "en"),
        ("Les comédiens jouent ce soir {} Lyon.", "à", "en"),
        ("Il est né {} Nantes.", "à", "en"),
        ("Mon ami vit {} Lisbonne depuis un an.", "à", "en"),
    ],
    # Pays au féminin : « en ».
    "pays_feminin": [
        ("Ma tante vit {} France depuis dix ans.", "en", "à"),
        ("Ils ont fait un long voyage {} Italie.", "en", "au"),
        ("Le peintre est parti {} Espagne.", "en", "à"),
        ("Mes grands-parents habitent {} Pologne.", "en", "au"),
        ("Le jeune homme fit ses études {} Allemagne.", "en", "à"),
        ("On cultive beaucoup de riz {} Chine.", "en", "au"),
        ("Le poète anglais est mort {} Grèce.", "en", "à"),
        ("Il a vécu longtemps {} Suède.", "en", "au"),
    ],
    # Pays masculin à initiale consonantique : « au ».
    "pays_masculin": [
        ("Ma cousine est partie vivre {} Japon.", "au", "en"),
        ("Le café pousse bien {} Brésil.", "au", "à"),
        ("Son neveu s'est installé {} Liban.", "au", "à"),
        ("Il fait très froid l'hiver {} Canada.", "au", "à le"),
        ("Les explorateurs sont arrivés {} Pérou.", "au", "à le"),
        ("Ce grand volcan se trouve {} Chili.", "au", "à"),
        ("Mon voisin a longtemps travaillé {} Cameroun.", "au", "à"),
    ],
    # Nom de lieu pluriel : « aux ».
    "pays_pluriel": [
        ("Ma sœur fait ses études {} États-Unis.", "aux", "en"),
        ("Les tulipes poussent partout {} Pays-Bas.", "aux", "en"),
        ("Ils passent leurs vacances {} Antilles.", "aux", "à les"),
        ("Son aïeul avait fait fortune {} Indes.", "aux", "à les"),
        ("Le missionnaire est parti {} Philippines.", "aux", "à"),
        ("Le navigateur fait escale {} Açores.", "aux", "à"),
    ],
    # Provenance : « de » / « d' » (pays féminin), « du » (masculin),
    # « des » (pluriel).
    "provenance": [
        ("Mon père revient {} Japon demain.", "du", "de"),
        ("Ces oranges viennent {}Espagne.", "d'", "du "),
        ("Elle rentre {} Norvège la semaine prochaine.", "de", "du"),
        ("Ce café vient {} Colombie.", "de", "du"),
        ("Les voyageurs arrivent {} États-Unis ce soir.", "des", "de les"),
        ("Mon oncle est revenu {} Portugal.", "du", "de le"),
        ("Ce marbre blanc vient {}Italie.", "d'", "du "),
        ("Ces fleurs viennent {} Pays-Bas.", "des", "de"),
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
    # Affichage rapide : python evals/paires/prepositions.py
    _paires = generate()
    for _p in _paires:
        print(f"[{_p['gabarit']}] {_p['good']}  |  *{_p['bad']}")
    _courtes = sum(len(p["good"]) < len(p["bad"]) for p in _paires)
    print(f"{len(_paires)} paires, bonne phrase plus courte : "
          f"{100 * _courtes / len(_paires):.1f} %")
