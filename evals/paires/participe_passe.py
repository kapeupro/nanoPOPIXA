"""Paires minimales : participe passé et auxiliaires.

Chaque paire oppose une phrase grammaticale (« good ») à une phrase
agrammaticale (« bad ») qui ne diffère QUE par un mot : l'auxiliaire, ou la
forme du participe. Chaque gabarit est un cadre de phrase avec une case « {} »
où l'on insère la forme correcte ou la forme fautive : tout le reste est
identique caractère pour caractère (différence minimale garantie par
construction).

Gabarits (identifiant court dans le champ « gabarit ») :
  - aux_etre : verbe qui se conjugue avec « être » (venir, arriver, partir,
    tomber, naître, devenir, aller, revenir)
        « Le facteur est arrivé ce matin. » / « *Le facteur a arrivé ce matin. »
  - aux_avoir : verbe qui se conjugue avec « avoir » (intransitifs sans
    passif possible, verbes météo, transitif suivi de son objet)
        « Il a plu toute la nuit. » / « *Il est plu toute la nuit. »
  - accord_genre : accord en genre du participe conjugué avec « être »
        « Ma tante est arrivée hier soir. » / « *Ma tante est arrivé hier soir. »
  - accord_nombre : accord en nombre du participe conjugué avec « être »
        « Mes amis sont partis sans moi. » / « *Mes amis sont parti sans moi. »
  - pronominal : verbes essentiellement pronominaux (s'évanouir, s'enfuir,
    s'envoler, se souvenir, s'agenouiller, s'emparer), dont le participe
    s'accorde toujours avec le sujet
        « Elle s'est évanouie de peur. » / « *Elle s'est évanoui de peur. »
  - passif : accord du participe à la voix passive
        « La maison a été construite par... » / « *... a été construit par... »
  - forme_irreguliere : participe irrégulier remplacé par une forme
    « régularisée » inexistante
        « Il a pris le train de nuit. » / « *Il a prendu le train de nuit. »
  - participe_infinitif : confusion du participe en -é et de l'infinitif
    en -er (participe après l'auxiliaire, infinitif après un semi-auxiliaire)
        « Elle a fermé la fenêtre. » / « *Elle a fermer la fenêtre. »
        « Je vais chercher du pain. » / « *Je vais cherché du pain. »

Exclusions volontaires (pour que la phrase fautive soit « sans conteste »
agrammaticale, et non simplement familière, régionale ou ancienne) :
  - pas d'accord du COD antéposé avec « avoir » (trop subtil, souvent violé) ;
    les pronominaux retenus sont essentiellement pronominaux, sans COD distinct ;
  - pas de verbes à double auxiliaire selon le sens ou l'usage (monter,
    descendre, sortir, rentrer, passer, retourner, demeurer, rester,
    disparaître, apparaître...) dans les gabarits d'auxiliaire ; ils
    n'apparaissent qu'avec « être » dans les gabarits d'accord, où
    l'auxiliaire est fixe ;
  - dans « aux_avoir », aucun participe qui puisse faire un passif ou un
    adjectif après « être » (« Il est couru », « Il est pleuré par... »,
    « Le gâteau est réussi ») ni de « il » impersonnel suivi d'un transitif
    (« Il est mangé beaucoup de pain » est un passif impersonnel correct) ;
  - pas de « nous » ni de « vous » dans les gabarits d'accord : le « nous » de
    modestie et le « vous » de politesse autorisent le singulier ;
  - formes régularisées fautives (« prendu », « ouvri »...) choisies pour
    n'être homographes d'aucun mot français (pas de « peigné », « voyé »...) ;
  - pas de participe employé aussi comme adjectif attribut de personne
    (« un homme très voyagé ») dans « aux_avoir ».

Équilibre des longueurs : la bonne phrase est strictement la plus courte dans
33 paires sur 68 (48,5 %), la plus longue dans 28, de même longueur dans 7.
Deux gabarits sont orientés par le phénomène lui-même, les formes de « être »
étant plus longues que celles de « avoir » (est / a, sont / ont,
sommes / avons) :
  - « aux_etre » : bonne phrase jamais la plus courte ; on y place les temps
    où les deux auxiliaires ont la même longueur (était / avait,
    seront / auront, êtes / avez) ;
  - « aux_avoir » : bonne phrase la plus courte, sauf avec avait / était et
    avez / êtes (longueurs égales).
Il en va de même dans « forme_irreguliere », où la forme régularisée fautive
est le plus souvent plus longue (« pris » / « *prendu ») : on y compense par
les participes en -ert (« ouvert » / « *ouvri »), plus longs que la forme
fautive, et par des formes de même longueur (« vécu » / « *vivé »). Les
autres gabarits sont équilibrés à parts égales (masculin singulier attendu =
bonne plus courte ; féminin ou pluriel attendu = bonne plus longue ;
participe attendu = bonne plus courte ; infinitif attendu = bonne plus longue).

Module autonome : bibliothèque standard uniquement, aucun aléatoire.
"""

PHENOMENE = "participe_passe"

# Gabarit -> liste de (cadre, forme correcte, forme fautive).
# Le cadre contient exactement une case « {} ».
GABARITS = {
    # Auxiliaire « être » attendu, « avoir » fautif (verbes de mouvement ou
    # de changement d'état, sans emploi transitif courant).
    "aux_etre": [
        ("Le facteur {} arrivé ce matin.", "est", "a"),
        ("Mon oncle {} venu nous voir dimanche.", "est", "a"),
        ("L'enfant {} tombé dans l'escalier.", "est", "a"),
        ("Victor Hugo {} né à Besançon.", "est", "a"),
        ("Son fils {} devenu médecin.", "est", "a"),
        ("Les invités {} allés au jardin.", "sont", "ont"),
        ("Le train {} parti depuis une heure.", "était", "avait"),
        ("Ils {} revenus avant la nuit.", "seront", "auront"),
        ("Vous {} arrivés trop tard.", "êtes", "avez"),
    ],
    # Auxiliaire « avoir » attendu, « être » fautif.
    "aux_avoir": [
        ("Il {} plu toute la nuit.", "a", "est"),
        ("Le chien {} aboyé contre les passants.", "a", "est"),
        ("Les enfants {} dormi dans la grange.", "ont", "sont"),
        ("Elle {} souri à son voisin.", "a", "est"),
        ("Nous {} marché jusqu'au village.", "avons", "sommes"),
        ("Vous {} beaucoup hésité avant de répondre.", "avez", "êtes"),
        ("Le vieillard {} toussé toute la soirée.", "avait", "était"),
        ("Il {} neigé sur la montagne.", "a", "est"),
        ("Ma sœur {} lu ce roman deux fois.", "a", "est"),
    ],
    # Accord en genre avec « être » (féminin attendu / masculin attendu).
    "accord_genre": [
        ("Ma tante est {} hier soir.", "arrivée", "arrivé"),
        ("Le boulanger est {} très tôt.", "parti", "partie"),
        ("La reine est {} en exil.", "morte", "mort"),
        ("Le petit garçon est {} du mur.", "tombé", "tombée"),
        ("Elle est {} dans un petit village.", "née", "né"),
        ("Il est {} seul à la fête.", "venu", "venue"),
        ("La jeune fille était {} sans manteau.", "sortie", "sorti"),
        ("Le vieux marin est {} en Bretagne.", "retourné", "retournée"),
    ],
    # Accord en nombre avec « être » (pluriel attendu / singulier attendu).
    "accord_nombre": [
        ("Les hirondelles sont {} au printemps.", "revenues", "revenue"),
        ("Mes amis sont {} sans moi.", "partis", "parti"),
        ("Le voyageur est {} du train.", "descendu", "descendus"),
        ("Les feuilles sont {} dans la cour.", "tombées", "tombée"),
        ("La lettre est {} avec un jour de retard.", "arrivée", "arrivées"),
        ("L'oiseau est {} de sa cage.", "sorti", "sortis"),
        ("Elles étaient {} pour le mariage.", "venues", "venue"),
        ("Mon père est {} au grenier.", "monté", "montés"),
    ],
    # Verbes essentiellement pronominaux : accord avec le sujet.
    "pronominal": [
        ("Elle s'est {} de peur.", "évanouie", "évanoui"),
        ("Le prisonnier s'est {} pendant la nuit.", "enfui", "enfuie"),
        ("Les pigeons se sont {}.", "envolés", "envolé"),
        ("Mon frère s'est {} de cette histoire.", "souvenu", "souvenue"),
        ("Les femmes se sont {} dans l'église.", "agenouillées", "agenouillée"),
        ("Le chat s'est {} du poisson.", "emparé", "emparée"),
    ],
    # Voix passive : le participe s'accorde avec le sujet.
    "passif": [
        ("La maison a été {} par mon grand-père.", "construite", "construit"),
        ("Le pont fut {} pendant la guerre.", "détruit", "détruite"),
        ("Ces lettres ont été {} par un poète.", "écrites", "écrite"),
        ("Le voleur a été {} par la police.", "arrêté", "arrêtée"),
        ("Les portes seront {} à minuit.", "fermées", "fermés"),
        ("Le repas est {} dans la grande salle.", "servi", "servis"),
    ],
    # Participe irrégulier / forme régularisée inexistante.
    "forme_irreguliere": [
        ("Il a {} le train de nuit.", "pris", "prendu"),
        ("J'ai {} mes devoirs avant le dîner.", "fait", "faisé"),
        ("Elle a {} sa robe bleue.", "mis", "mettu"),
        ("Le notaire a {} une longue lettre.", "écrit", "écrivé"),
        ("Nous avons {} un verre d'eau.", "bu", "buvé"),
        ("Tu as {} un joli tableau.", "peint", "peindu"),
        ("Elle a {} un bouquet de fleurs.", "reçu", "recevu"),
        ("Le maître a {} la vérité.", "dit", "disé"),
        ("Le marchand a {} sa boutique.", "ouvert", "ouvri"),
        ("Ma mère nous a {} des cadeaux.", "offert", "offri"),
        ("Le peuple a beaucoup {} de la faim.", "souffert", "souffri"),
        ("La neige a {} les champs.", "couvert", "couvri"),
        ("Ma grand-mère a {} à la campagne.", "vécu", "vivé"),
        ("Le serpent a {} le berger.", "mordu", "mordi"),
    ],
    # Participe en -é (après l'auxiliaire) / infinitif en -er (après un
    # semi-auxiliaire ou un verbe de volonté).
    "participe_infinitif": [
        ("Le renard a {} la poule.", "mangé", "manger"),
        ("Nous avons {} tout l'été.", "travaillé", "travailler"),
        ("Elle a {} la fenêtre.", "fermé", "fermer"),
        ("Le jardinier a {} les roses.", "coupé", "couper"),
        ("Il faut {} la vaisselle.", "laver", "lavé"),
        ("Je vais {} du pain.", "chercher", "cherché"),
        ("Elle veut {} au directeur.", "parler", "parlé"),
        ("Nous allons {} le château.", "visiter", "visité"),
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
    # Affichage rapide : python evals/paires/participe_passe.py
    _paires = generate()
    for _p in _paires:
        print(f"[{_p['gabarit']}] {_p['good']}  |  *{_p['bad']}")
    _courtes = sum(len(p["good"]) < len(p["bad"]) for p in _paires)
    print(f"{len(_paires)} paires, bonne phrase plus courte : "
          f"{100 * _courtes / len(_paires):.1f} %")
