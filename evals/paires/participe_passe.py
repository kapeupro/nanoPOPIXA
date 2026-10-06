"""Paires minimales : participe passé et auxiliaires.

Chaque paire oppose une phrase grammaticale (« good ») à une phrase
agrammaticale (« bad ») qui ne diffère QUE par un mot : l'auxiliaire, ou la
forme du participe. Chaque gabarit est un cadre de phrase avec une case « {} »
où l'on insère la forme correcte ou la forme fautive : tout le reste est
identique caractère pour caractère (différence minimale garantie par
construction).

Gabarits (identifiant court dans le champ « gabarit ») :
  - aux_etre : verbe qui se conjugue avec « être » (venir, revenir, devenir,
    arriver, aller, naître, mourir), à des temps variés (passé composé,
    plus-que-parfait, futur antérieur, conditionnel passé, passé antérieur,
    infinitif passé)
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
  - participe_infinitif : participe attendu après l'auxiliaire, remplacé par
    l'infinitif homophone ou quasi homophone (après « avoir » : -é / -er ;
    après « être » au féminin pluriel : -ées / -er)
        « Elle a fermé la fenêtre. » / « *Elle a fermer la fenêtre. »
        « Les lettres sont arrivées ce matin. » / « *... sont arriver ... »
    Aucun contexte où l'infinitif est la bonne forme (« Il faut laver... ») :
    ce serait un test d'orthographe de l'infinitif, hors du phénomène.

Une seule règle violée par paire (gabarits d'auxiliaire) : le sujet y est
toujours masculin singulier (nom, « il », « tu »). Avec « être », le
participe a alors la même forme qu'avec « avoir » ; la phrase fautive ne viole
donc que le choix de l'auxiliaire, jamais l'accord (on évite « *Ils ont
venus », « *Elle est souri », « *Nous sommes marché »...).

Exclusions volontaires (pour que la phrase fautive soit « sans conteste »
agrammaticale, et non simplement familière, régionale ou ancienne) :
  - pas d'accord du COD antéposé avec « avoir » (trop subtil, souvent violé) ;
    les pronominaux retenus sont essentiellement pronominaux, sans COD distinct ;
  - dans « aux_etre », on écarte les verbes dont l'emploi avec « avoir » est
    répandu dans l'usage populaire ou régional ou attesté dans la langue
    classique (tomber, partir, rester...) ; on privilégie venir, devenir,
    naître, mourir (aucun emploi avec « avoir »), arriver et aller restant
    limités à 3 paires ; pas non plus de verbe à double auxiliaire selon le
    sens (monter, descendre, sortir, rentrer, passer, retourner, demeurer,
    disparaître, apparaître...) ; ces
    verbes n'apparaissent qu'avec « être » dans les gabarits d'accord, où
    l'auxiliaire est fixe ;
  - dans « aux_avoir », aucun participe qui puisse faire un passif ou un
    adjectif après « être » (« Il est couru », « Il est pleuré par... »,
    « Le gâteau est réussi »), sauf un transitif suivi de son objet direct
    (« *Le professeur est lu ce roman »), où la lecture passive est exclue ;
    pas de « il » impersonnel suivi d'un transitif (« Il est mangé beaucoup
    de pain » est un passif impersonnel correct) ;
  - pas de « nous » ni de « vous » sujets d'un participe conjugué avec
    « être » : le « nous » de modestie et le « vous » de politesse autorisent
    le singulier, le pluriel ordinaire exige l'accord ;
  - formes régularisées fautives (« prendu », « mettu »...) choisies pour
    n'être homographes d'aucun mot français (pas de « peigné », « voyé »...) ;
    une seule forme en *-ri (« ouvri ») pour éviter un schéma répétitif ;
  - pas de participe employé aussi comme adjectif attribut de personne
    (« un homme très voyagé ») dans « aux_avoir ».

Équilibre des longueurs (en caractères) : la bonne phrase est strictement la
plus courte dans 33 paires sur 69 (47,8 %), la plus longue dans 24, de même
longueur dans 12. Deux gabarits sont orientés par le phénomène lui-même, les
formes de « être » étant le plus souvent plus longues que celles de « avoir »
(est / a) :
  - « aux_etre » : bonne phrase plus longue dans 5 paires sur 12 (est / a) ;
    pour limiter cet indice, 6 paires utilisent des formes de même longueur
    (était / avait, sera / aura, es / as, serait / aurait, fut / eut) et 1 a
    la bonne phrase plus courte (être / avoir) ;
  - « aux_avoir » : bonne phrase la plus courte dans 8 paires sur 12 (a / est),
    de même longueur dans les 4 autres (as / es, avait / était, eut / fut,
    aurait / serait).
Dans « forme_irreguliere », la forme régularisée fautive est le plus souvent
plus longue (« pris » / « *prendu ») : 6 paires sur 9 ont la bonne phrase plus
courte, 2 ont des formes de même longueur (« vécu » / « *vivé »,
« couru » / « *couri »), 1 a la bonne phrase plus longue (« ouvert » /
« *ouvri »). Les autres gabarits sont équilibrés à parts égales (masculin
singulier attendu = bonne plus courte ; féminin ou pluriel attendu = bonne
plus longue ; participe en -é attendu = bonne plus courte ; participe en -ées
attendu = bonne plus longue).

Biais en nombre de tokens (à signaler lors de l'analyse) : dans
« forme_irreguliere », la forme fautive est un non-mot que le tokeniseur BPE
découpe en plus de morceaux que le participe existant ; la somme des
log-probabilités pénalise alors la mauvaise phrase indépendamment de toute
connaissance grammaticale. Ce gabarit est donc réduit à 9 paires (13 % du
jeu) et ses scores doivent être lus à part. De façon générale, publier les
scores PAR GABARIT (champ « gabarit ») plutôt que le seul score global.

Module autonome : bibliothèque standard uniquement, aucun aléatoire.
"""

PHENOMENE = "participe_passe"

# Gabarit -> liste de (cadre, forme correcte, forme fautive).
# Le cadre contient exactement une case « {} ».
GABARITS = {
    # Auxiliaire « être » attendu, « avoir » fautif. Sujet masculin singulier
    # (aucun accord en jeu) ; verbes sans emploi attesté avec « avoir ».
    "aux_etre": [
        ("Le facteur {} arrivé ce matin.", "est", "a"),
        ("Mon oncle {} venu nous voir dimanche.", "est", "a"),
        ("Le vieux chien {} mort cet hiver.", "est", "a"),
        ("Victor Hugo {} né à Besançon.", "est", "a"),
        ("Son fils {} devenu médecin.", "est", "a"),
        ("Mon cousin {} allé au jardin.", "était", "avait"),
        ("Le train {} arrivé depuis une heure.", "était", "avait"),
        ("Il {} revenu avant la nuit.", "sera", "aura"),
        ("Tu {} venu trop tard.", "es", "as"),
        ("Il {} mort sans ton aide.", "serait", "aurait"),
        ("Après {} devenu roi, il oublia ses amis.", "être", "avoir"),
        ("Dès qu'il {} revenu, la fête commença.", "fut", "eut"),
    ],
    # Auxiliaire « avoir » attendu, « être » fautif. Sujet masculin singulier :
    # avec « être », le participe garderait la même forme (pas d'accord en jeu).
    "aux_avoir": [
        ("Il {} plu toute la nuit.", "a", "est"),
        ("Le chien {} aboyé contre les passants.", "a", "est"),
        ("Le fermier {} dormi dans la grange.", "a", "est"),
        ("Il {} souri à son voisin.", "a", "est"),
        ("Le soldat {} marché jusqu'au village.", "a", "est"),
        ("Tu {} beaucoup hésité avant de répondre.", "as", "es"),
        ("Le vieillard {} toussé toute la soirée.", "avait", "était"),
        ("Il {} neigé sur la montagne.", "a", "est"),
        ("Le professeur {} lu ce roman deux fois.", "a", "est"),
        ("Quand il {} dîné, il sortit.", "eut", "fut"),
        ("Il {} ri de cette histoire.", "aurait", "serait"),
        ("Ce roi {} régné quarante ans.", "a", "est"),
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
    # Participe irrégulier / forme régularisée inexistante (gabarit réduit :
    # biais en nombre de tokens, voir la docstring).
    "forme_irreguliere": [
        ("Il a {} le train de nuit.", "pris", "prendu"),
        ("J'ai {} mes devoirs avant le dîner.", "fait", "faisé"),
        ("Elle a {} sa robe bleue.", "mis", "mettu"),
        ("Le notaire a {} une longue lettre.", "écrit", "écrivé"),
        ("Nous avons {} un verre d'eau.", "bu", "buvé"),
        ("Elle a {} un bouquet de fleurs.", "reçu", "recevu"),
        ("Le marchand a {} sa boutique.", "ouvert", "ouvri"),
        ("Ma grand-mère a {} à la campagne.", "vécu", "vivé"),
        ("Le garçon a {} jusqu'à la gare.", "couru", "couri"),
    ],
    # Participe attendu après l'auxiliaire ; l'infinitif est fautif.
    "participe_infinitif": [
        ("Le renard a {} la poule.", "mangé", "manger"),
        ("Nous avons {} tout l'été.", "travaillé", "travailler"),
        ("Elle a {} la fenêtre.", "fermé", "fermer"),
        ("Le jardinier a {} les roses.", "coupé", "couper"),
        ("Les fenêtres sont {} depuis hier.", "fermées", "fermer"),
        ("Les cerises sont {} de l'arbre.", "tombées", "tomber"),
        ("Les voisines sont {} nous voir.", "passées", "passer"),
        ("Les lettres sont {} ce matin.", "arrivées", "arriver"),
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
