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
  - contraction_a : à + le -> au, à + les -> aux ; et, à l'inverse, pas de
    contraction devant un nom féminin ou devant une voyelle
        « Ma mère va au marché le samedi. » / « *... à le marché ... »
        « Elle attend son père à la gare. » / « *... au gare. »
  - contraction_de : de + le -> du, de + les -> des (complément du nom ou du
    verbe) ; pas de contraction devant un féminin ou une voyelle
        « La maison des voisins a un grand jardin. » / « *... de les voisins ... »
        « La porte de l'église reste ouverte. » / « *La porte du église ... »
  - partitif : article partitif (du, de la, de l') et indéfini pluriel (des)
        « Le boulanger vend du pain frais. » / « *... de le pain frais. »
  - ville : « à » devant un nom de ville, « au » devant les villes dont le
    nom contient l'article « le »
        « Mon oncle habite à Paris ... » / « *... en Paris ... »
        « Il est né au Havre. » / « *Il est né à Le Havre. »
  - pays_feminin : « en » devant un pays ou une région au féminin
        « Ma tante vit en France ... » / « *... à France ... »
        « ... un long voyage en Italie. » / « *... au Italie. »
  - pays_masculin : « au » devant un pays masculin commençant par une consonne
        « Ma cousine est partie vivre au Japon. » / « *... en Japon. »
  - pays_pluriel : « aux » devant un nom de pays pluriel
        « Ma sœur fait ses études aux États-Unis. » / « *... en États-Unis. »
  - provenance : « de » (d') devant un pays féminin, « du » / « des » devant
    un pays masculin ou pluriel
        « Mon père revient du Japon demain. » / « *... de Japon demain. »

Exclusions volontaires (pour que la phrase fautive soit « sans conteste »
agrammaticale, et non simplement ancienne, régionale ou défendable) :
  - pas de « en » fautif devant Portugal, Danemark ou Canada : « en Portugal »,
    « en Danemark », « en Canada » se lisent dans les textes classiques
    (XVIIe-XVIIIe siècles) ; le Canada n'apparaît qu'avec « à le » fautif ;
  - pas de ville à initiale vocalique (« en Avignon », « en Arles » sont
    admis), ni de ville homonyme d'une région ou d'un département
    (Vienne, Québec...) ;
  - jamais « au » ou « du » fautif devant un nom de ville : « au Paris des
    années folles », « du Paris d'autrefois » sont corrects avec un
    complément, et « le Paris » peut désigner un café, un navire ou un club ;
    le seul fautif employé devant une ville est « en » ;
  - « France » est aussi un prénom (« Il a écrit à France ») : dans les
    gabarits de lieu, la case est toujours un complément circonstanciel de
    lieu (habiter, vivre, partir, se trouver, naître...), jamais après un
    verbe à complément d'attribution (« écrire à », « parler à »,
    « enseigner à ») ni après « voir en X » (« voir en Paris la capitale du
    monde » est correct) ;
  - « à le » / « de les » ne sont fautifs que devant un nom : aucun
    infinitif ne suit (« Il pense à le faire », « Il décide de les voir »
    sont corrects) ;
  - pas de nom à « h » aspiré (« du hibou », « au héros »), qui relève du
    phénomène « elision », ni de négation (« pas de pain »), qui change
    l'article partitif.

Équilibre des longueurs : la bonne phrase est strictement la plus courte dans
29 paires sur 62 (46,8 %), la plus longue dans 23, de même longueur dans 10.
La contraction raccourcit toujours la forme (« au » / « à le », « des » /
« de les ») ; chaque gabarit de contraction alterne donc des paires où la
contraction est la bonne forme (bonne phrase plus courte) et des paires où
elle est fautive (« à la gare » / « *au gare », bonne phrase plus longue).
Deux gabarits sont orientés par le phénomène lui-même :
  - « ville » : bonne phrase toujours la plus courte (« à » a une lettre,
    « en » deux ; « au » est plus court que « à Le ») ;
  - « pays_feminin » : bonne phrase jamais la plus courte (« en » contre
    « à », plus court, ou « au », de même longueur) ;
ils se compensent (8 paires chacun). « pays_masculin », « pays_pluriel » et
« provenance » mêlent les trois cas en variant la forme fautive (« en »,
« à », « à le », « de »...).

Module autonome, bibliothèque standard uniquement, sortie déterministe.
"""

PHENOMENE = "prepositions"

# Gabarit -> liste de (cadre, forme correcte, forme fautive).
# Le cadre contient exactement une case « {} ». Pour les formes élidées, la
# case est collée au mot suivant : la forme non élidée porte alors son espace.
GABARITS = {
    # Préposition « à » + article défini : contraction obligatoire devant
    # « le » / « les », impossible devant « la » et devant une voyelle.
    "contraction_a": [
        ("Ma mère va {} marché le samedi.", "au", "à le"),
        ("Le maître parle {} enfants avec douceur.", "aux", "à les"),
        ("Nous pensons souvent {} voyage de l'été dernier.", "au", "à le"),
        ("Le vieillard donne du pain {} pauvres.", "aux", "à les"),
        ("Les enfants jouent {} balle dans la cour.", "à la", "au"),
        ("Elle attend son père {} gare.", "à la", "au"),
        ("Il a attaché son cheval {}arbre.", "à l'", "au "),
        ("Mon frère entre {}école en septembre.", "à l'", "au "),
    ],
    # Préposition « de » + article défini (complément du nom ou du verbe).
    "contraction_de": [
        ("La maison {} voisins a un grand jardin.", "des", "de les"),
        ("Le bruit {} moulin réveille le village.", "du", "de le"),
        ("On entend le chant {} oiseaux au printemps.", "des", "de les"),
        ("Le fils {} roi monte un cheval blanc.", "du", "de le"),
        ("Nous marchons au bord {} mer.", "de la", "du"),
        ("La porte {}église reste ouverte.", "de l'", "du "),
        ("Le vieux soldat parle souvent {} guerre.", "de la", "du"),
        ("La lumière {}aube entre dans la chambre.", "de l'", "du "),
    ],
    # Article partitif (du, de la, de l') et indéfini pluriel (des).
    "partitif": [
        ("Le boulanger vend {} pain frais.", "du", "de le"),
        ("Les écoliers mangent {} pommes au goûter.", "des", "de les"),
        ("Mon grand-père boit {} vin au dîner.", "du", "de le"),
        ("Elle met {} sucre dans son café.", "du", "de le"),
        ("Il faut acheter {} farine pour le gâteau.", "de la", "du"),
        ("Le voyageur boit {}eau à la fontaine.", "de l'", "du "),
        ("Nous avons eu {} chance cette année.", "de la", "du"),
        ("Elle verse {}huile dans la poêle.", "de l'", "du "),
    ],
    # Nom de ville : « à » (jamais « en ») ; « au » si le nom contient « le ».
    "ville": [
        ("Mon oncle habite {} Paris depuis longtemps.", "à", "en"),
        ("Nous passerons l'hiver {} Rome.", "à", "en"),
        ("Elle a étudié la peinture {} Venise.", "à", "en"),
        ("Le navire arrive {} Marseille demain matin.", "à", "en"),
        ("Mon cousin travaille {} Londres.", "à", "en"),
        ("Les comédiens jouent ce soir {} Lyon.", "à", "en"),
        ("Il est né {} Havre.", "au", "à Le"),
        ("Mon ami vit {} Caire depuis un an.", "au", "à Le"),
    ],
    # Pays ou région au féminin : « en ».
    "pays_feminin": [
        ("Ma tante vit {} France depuis dix ans.", "en", "à"),
        ("Ils ont fait un long voyage {} Italie.", "en", "au"),
        ("Le peintre est parti {} Espagne.", "en", "à"),
        ("Mes grands-parents habitent {} Bretagne.", "en", "au"),
        ("Le jeune homme fit ses études {} Allemagne.", "en", "à"),
        ("On cultive beaucoup de riz {} Chine.", "en", "au"),
        ("Le poète anglais est mort {} Grèce.", "en", "à"),
        ("Il a vécu longtemps {} Angleterre.", "en", "au"),
    ],
    # Pays masculin à initiale consonantique : « au ».
    "pays_masculin": [
        ("Ma cousine est partie vivre {} Japon.", "au", "en"),
        ("Le café pousse bien {} Brésil.", "au", "à"),
        ("Nous irons {} Mexique l'an prochain.", "au", "en"),
        ("Son neveu s'est installé {} Maroc.", "au", "à"),
        ("Il fait très froid l'hiver {} Canada.", "au", "à le"),
        ("Les explorateurs sont arrivés {} Pérou.", "au", "à le"),
        ("Mon voisin a longtemps vécu {} Sénégal.", "au", "en"),
        ("Ce grand volcan se trouve {} Chili.", "au", "à"),
    ],
    # Nom de pays pluriel : « aux ».
    "pays_pluriel": [
        ("Ma sœur fait ses études {} États-Unis.", "aux", "en"),
        ("Les tulipes poussent partout {} Pays-Bas.", "aux", "en"),
        ("Ils passent leurs vacances {} Antilles.", "aux", "à les"),
        ("Mon parrain a émigré {} États-Unis.", "aux", "à les"),
        ("Les deux frères vivent {} Pays-Bas.", "aux", "à les"),
        ("Le missionnaire est parti {} Philippines.", "aux", "en"),
    ],
    # Provenance : « de » / « d' » (pays féminin), « du » (masculin),
    # « des » (pluriel).
    "provenance": [
        ("Mon père revient {} Japon demain.", "du", "de"),
        ("Ces oranges viennent {}Espagne.", "d'", "du "),
        ("Elle rentre {} France la semaine prochaine.", "de", "du"),
        ("Ce thé vient {} Chine.", "de", "du"),
        ("Les voyageurs arrivent {} États-Unis ce soir.", "des", "de les"),
        ("Mon oncle est revenu {} Mexique.", "du", "de le"),
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
