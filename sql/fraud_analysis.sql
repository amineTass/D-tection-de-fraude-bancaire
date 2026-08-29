-- =====================================================================
--  Analyse de fraude bancaire - requêtes SQLite sur la table "transactions"
-- ---------------------------------------------------------------------
--  Prérequis : créer une table avec la requête au début du fichier,
--  puis importer creditcard.csv (via sqlite3, DB Browser ou pandas).
--
--  Table cible :
--    CREATE TABLE transactions (
--        id      INTEGER PRIMARY KEY,
--        Time    REAL,
--        V1..V28 REAL,
--        Amount  REAL,
--        Class   INTEGER        -- 1 = fraude, 0 = transaction normale
--    );
-- =====================================================================

-- ---------------------------------------------------------------------
-- 1. Répartition fraude / normale
--    Compte le nombre de transactions par classe pour mesurer
--    le déséquilibre du jeu de données (~0,17 % de fraudes attendu).
-- ---------------------------------------------------------------------
SELECT
    Class,
    COUNT(*)                       AS nombre_transactions,
    ROUND(COUNT(*) * 100.0 /
          (SELECT COUNT(*) FROM transactions), 2) AS pourcentage
FROM transactions
GROUP BY Class
ORDER BY Class;

-- ---------------------------------------------------------------------
-- 2. Montants moyens, minimums et maximums par classe
--    Compare les statistiques de montant entre fraudes et normales
--    (souvent : fraudes de montants élevés mais aussi beaucoup de petites).
-- ---------------------------------------------------------------------
SELECT
    Class,
    AVG(Amount)  AS montant_moyen,
    MIN(Amount)  AS montant_min,
    MAX(Amount)  AS montant_max,
    MEDIAN(Amount) AS montant_median
FROM transactions
GROUP BY Class;

-- ---------------------------------------------------------------------
-- 3. Répartition horaire des fraudes
--    Time est le nombre de secondes écoulées depuis la première
--    transaction. On le convertit en heure (0-23) avec MOD(Time/3600, 24).
--    Met en évidence les créneaux à risque (ex. la nuit).
-- ---------------------------------------------------------------------
SELECT
    CAST(MOD(Time / 3600, 24) AS INTEGER) AS heure,
    COUNT(*)                              AS nombre_fraudes
FROM transactions
WHERE Class = 1
GROUP BY heure
ORDER BY heure;

-- ---------------------------------------------------------------------
-- 4. Top 10 des fraudes par montant
--    Liste les transactions frauduleuses les plus élevées, avec leur
--    instant (en heures) et leur montant, triées par montant décroissant.
-- ---------------------------------------------------------------------
SELECT
    id,
    ROUND(Time / 3600, 2) AS temps_heures,
    Amount                AS montant,
    Class
FROM transactions
WHERE Class = 1
ORDER BY Amount DESC
LIMIT 10;

-- ---------------------------------------------------------------------
-- 5. Taux de fraude par tranche de montant
--    Observe si la fraude est concentrée sur certaines tranches.
--    (CASE WHEN permet de définir des buckets, ici en EUR.)
-- ---------------------------------------------------------------------
SELECT
    CASE
        WHEN Amount < 50       THEN '0-50'
        WHEN Amount < 100      THEN '50-100'
        WHEN Amount < 500      THEN '100-500'
        WHEN Amount < 1000     THEN '500-1000'
        ELSE                        '1000+'
    END                          AS tranche_montant,
    COUNT(*)                     AS total,
    SUM(Class)                   AS fraudes,
    ROUND(SUM(Class) * 100.0 / COUNT(*), 3) AS taux_fraude_pct
FROM transactions
GROUP BY tranche_montant
ORDER BY MIN(Amount);

-- ---------------------------------------------------------------------
-- 6. Durée moyenne entre deux transactions frauduleuses par compte
--    (si une colonne de compte existait). Conservé ici comme requête
--    indicative pour un schéma enrichi avec user_id / card_id.
-- ---------------------------------------------------------------------
-- SELECT
--     card_id,
--     COUNT(*)                                  AS nb_fraudes,
--     MAX(Time) - MIN(Time) AS duree_ecran,
--     ROUND((MAX(Time) - MIN(Time)) / NULLIF(COUNT(*), 0), 2) AS duree_moyenne
-- FROM transactions
-- WHERE Class = 1
-- GROUP BY card_id
-- HAVING COUNT(*) > 1
-- ORDER BY COUNT(*) DESC;