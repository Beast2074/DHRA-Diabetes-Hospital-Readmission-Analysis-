-- business questions answered in SQL
-- each block starts with "-- name:" so python can run them one by one
-- @readmit_cost is set by python from config.py (15200 by default)

-- name: q01_overall_kpis
-- question: how big is the readmission problem in this hospital network?
SELECT
    k.*,
    k.readmissions_30d * @readmit_cost AS readmission_cost_usd
FROM vw_kpi_summary k;


-- name: q02_readmit_by_age
-- question: which age groups get readmitted the most?
SELECT
    age_group,
    COUNT(*)                                        AS encounters,
    ROUND(COUNT(*) * 100 / SUM(COUNT(*)) OVER (), 2) AS pct_of_encounters,
    SUM(readmit_30)                                 AS readmissions,
    ROUND(AVG(readmit_30) * 100, 2)                 AS readmit_rate_pct
FROM vw_eligible_encounters
GROUP BY age_group
ORDER BY age_group;


-- name: q03_prior_admissions_dose_response
-- question: does risk climb with the number of inpatient stays in the past year?
WITH b AS (
    SELECT LEAST(number_inpatient, 5) AS bucket, readmit_30
    FROM vw_eligible_encounters
)
SELECT
    CASE WHEN bucket = 5 THEN '5+' ELSE CAST(bucket AS CHAR) END AS prior_inpatient_visits,
    COUNT(*)                        AS encounters,
    SUM(readmit_30)                 AS readmissions,
    ROUND(AVG(readmit_30) * 100, 2) AS readmit_rate_pct,
    ROUND(AVG(readmit_30) / FIRST_VALUE(AVG(readmit_30)) OVER (ORDER BY bucket), 2) AS relative_risk_vs_zero
FROM b
GROUP BY bucket
ORDER BY bucket;


-- name: q04_discharge_destination
-- question: where patients go after discharge vs readmission, and the extra cost vs going home
WITH d AS (
    SELECT
        discharge_group,
        COUNT(*)         AS encounters,
        SUM(readmit_30)  AS readmissions,
        AVG(readmit_30)  AS rate
    FROM vw_eligible_encounters
    GROUP BY discharge_group
),
home AS (
    SELECT rate AS home_rate FROM d WHERE discharge_group = 'Home'
)
SELECT
    d.discharge_group,
    d.encounters,
    d.readmissions,
    ROUND(d.rate * 100, 2)                                        AS readmit_rate_pct,
    ROUND((d.rate - h.home_rate) * d.encounters)                  AS excess_readmissions_vs_home,
    ROUND((d.rate - h.home_rate) * d.encounters * @readmit_cost)  AS excess_cost_usd
FROM d
CROSS JOIN home h
ORDER BY readmit_rate_pct DESC;


-- name: q05_primary_diagnosis_cost
-- question: which primary diagnosis groups cost the most in readmissions?
SELECT
    d.diag_group                          AS primary_diagnosis,
    COUNT(*)                              AS encounters,
    SUM(v.readmit_30)                     AS readmissions,
    ROUND(AVG(v.readmit_30) * 100, 2)     AS readmit_rate_pct,
    SUM(v.readmit_30) * @readmit_cost     AS readmission_cost_usd,
    RANK() OVER (ORDER BY SUM(v.readmit_30) DESC) AS cost_rank
FROM vw_eligible_encounters v
JOIN encounter_diagnoses d
  ON d.encounter_id = v.encounter_id AND d.diag_position = 1
GROUP BY d.diag_group
ORDER BY cost_rank;


-- name: q06_hba1c_care_gap
-- question: how often is HbA1c tested, and does it differ by primary diagnosis?
SELECT
    d.diag_group                                                         AS primary_diagnosis,
    COUNT(*)                                                             AS encounters,
    ROUND(AVG(v.a1c_result <> 'Not Tested') * 100, 2)                    AS hba1c_tested_pct,
    ROUND(AVG(CASE WHEN v.a1c_result <> 'Not Tested' THEN v.readmit_30 END) * 100, 2) AS readmit_rate_tested,
    ROUND(AVG(CASE WHEN v.a1c_result =  'Not Tested' THEN v.readmit_30 END) * 100, 2) AS readmit_rate_not_tested
FROM vw_eligible_encounters v
JOIN encounter_diagnoses d
  ON d.encounter_id = v.encounter_id AND d.diag_position = 1
GROUP BY d.diag_group
HAVING COUNT(*) >= 1000
ORDER BY encounters DESC;


-- name: q07_hba1c_result_and_action
-- question: same 4 groups as the Strack paper - test result and whether meds were changed
SELECT
    CASE
        WHEN a1c_result = 'Not Tested'                    THEN '1. Not tested'
        WHEN a1c_result IN ('>7', '>8') AND med_change = 1 THEN '2. High result, meds changed'
        WHEN a1c_result IN ('>7', '>8')                    THEN '3. High result, no change'
        ELSE '4. Normal result'
    END                             AS hba1c_group,
    COUNT(*)                        AS encounters,
    SUM(readmit_30)                 AS readmissions,
    ROUND(AVG(readmit_30) * 100, 2) AS readmit_rate_pct
FROM vw_first_encounters
GROUP BY hba1c_group
ORDER BY hba1c_group;


-- name: q08_specialty_ranking
-- question: which admitting specialties have the highest readmission rate (min 1,000 stays)?
SELECT
    medical_specialty,
    COUNT(*)                        AS encounters,
    ROUND(AVG(readmit_30) * 100, 2) AS readmit_rate_pct,
    DENSE_RANK() OVER (ORDER BY AVG(readmit_30) DESC) AS risk_rank
FROM vw_eligible_encounters
WHERE medical_specialty <> 'Unknown'
GROUP BY medical_specialty
HAVING COUNT(*) >= 1000
ORDER BY risk_rank;


-- name: q09_visit_sequence
-- question: are later visits of the same patient riskier? (encounter_id order used as time order)
WITH seq AS (
    SELECT
        readmit_30,
        LEAST(ROW_NUMBER() OVER (PARTITION BY patient_nbr ORDER BY encounter_id), 4) AS visit_bucket
    FROM vw_eligible_encounters
)
SELECT
    CASE WHEN visit_bucket = 4 THEN '4+' ELSE CAST(visit_bucket AS CHAR) END AS visit_number,
    COUNT(*)                        AS encounters,
    ROUND(AVG(readmit_30) * 100, 2) AS readmit_rate_pct
FROM seq
GROUP BY visit_bucket
ORDER BY visit_bucket;


-- name: q10_frequent_flyers
-- question: how much of the readmission load comes from a small group of repeat patients?
WITH p AS (
    SELECT
        patient_nbr,
        COUNT(*)        AS stays,
        SUM(readmit_30) AS readmits
    FROM vw_eligible_encounters
    GROUP BY patient_nbr
)
SELECT
    CASE WHEN stays >= 3 THEN '3+ stays' WHEN stays = 2 THEN '2 stays' ELSE '1 stay' END AS patient_type,
    COUNT(*)                                            AS patients,
    ROUND(COUNT(*) * 100 / SUM(COUNT(*)) OVER (), 2)    AS pct_of_patients,
    SUM(stays)                                          AS encounters,
    ROUND(SUM(stays) * 100 / SUM(SUM(stays)) OVER (), 2) AS pct_of_encounters,
    SUM(readmits)                                       AS readmissions,
    ROUND(SUM(readmits) * 100 / SUM(SUM(readmits)) OVER (), 2) AS pct_of_readmissions
FROM p
GROUP BY patient_type
ORDER BY patient_type;


-- name: q11_top_primary_codes_pareto
-- question: which exact ICD-9 codes drive readmissions (running total = pareto)?
WITH codes AS (
    SELECT
        d.icd9_code,
        d.diag_group,
        COUNT(*)          AS encounters,
        SUM(v.readmit_30) AS readmissions
    FROM vw_eligible_encounters v
    JOIN encounter_diagnoses d
      ON d.encounter_id = v.encounter_id AND d.diag_position = 1
    GROUP BY d.icd9_code, d.diag_group
),
ranked AS (
    SELECT
        c.*,
        ROW_NUMBER() OVER (ORDER BY readmissions DESC, icd9_code) AS code_rank,
        SUM(readmissions) OVER (ORDER BY readmissions DESC, icd9_code ROWS UNBOUNDED PRECEDING) AS running_readmissions,
        SUM(readmissions) OVER () AS total_readmissions,
        COUNT(*) OVER () AS total_codes
    FROM codes c
)
SELECT
    code_rank,
    icd9_code,
    diag_group,
    encounters,
    readmissions,
    ROUND(readmissions * 100 / encounters, 2)                 AS readmit_rate_pct,
    ROUND(running_readmissions * 100 / total_readmissions, 2) AS cumulative_pct_of_readmissions,
    total_codes
FROM ranked
WHERE code_rank <= 15
ORDER BY code_rank;


-- name: q12_insulin_dosage
-- question: does an insulin dose change during the stay signal higher risk?
SELECT
    COALESCE(em.dosage_status, 'Not prescribed') AS insulin_status,
    COUNT(*)                                     AS encounters,
    ROUND(AVG(v.readmit_30) * 100, 2)            AS readmit_rate_pct
FROM vw_eligible_encounters v
LEFT JOIN encounter_medications em
  ON em.encounter_id = v.encounter_id
 AND em.medication_id = (SELECT medication_id FROM medications WHERE medication_name = 'insulin')
GROUP BY insulin_status
ORDER BY readmit_rate_pct DESC;


-- name: q13_polypharmacy
-- question: number of distinct medications given vs readmission
SELECT
    CASE
        WHEN num_medications < 10 THEN '1. under 10'
        WHEN num_medications < 20 THEN '2. 10-19'
        WHEN num_medications < 30 THEN '3. 20-29'
        ELSE '4. 30+'
    END                             AS medication_count,
    COUNT(*)                        AS encounters,
    ROUND(AVG(readmit_30) * 100, 2) AS readmit_rate_pct
FROM vw_eligible_encounters
GROUP BY medication_count
ORDER BY medication_count;


-- name: q14_length_of_stay
-- question: length of stay vs readmission and bed days used
SELECT
    CASE
        WHEN time_in_hospital <= 2 THEN '1. 1-2 days'
        WHEN time_in_hospital <= 4 THEN '2. 3-4 days'
        WHEN time_in_hospital <= 7 THEN '3. 5-7 days'
        ELSE '4. 8-14 days'
    END                             AS length_of_stay,
    COUNT(*)                        AS encounters,
    SUM(time_in_hospital)           AS bed_days,
    ROUND(AVG(readmit_30) * 100, 2) AS readmit_rate_pct
FROM vw_eligible_encounters
GROUP BY length_of_stay
ORDER BY length_of_stay;


-- name: q15_admission_source
-- question: do ER admissions come back more than referrals or transfers?
SELECT
    admission_source,
    COUNT(*)                        AS encounters,
    ROUND(AVG(readmit_30) * 100, 2) AS readmit_rate_pct
FROM vw_eligible_encounters
GROUP BY admission_source
ORDER BY encounters DESC;


-- name: q16_lace_tier_check
-- question: does the standard LACE tiering (via the stored function) separate risk on this data?
SELECT
    fn_lace_risk_tier(lace_score)   AS lace_tier,
    COUNT(*)                        AS encounters,
    ROUND(COUNT(*) * 100 / SUM(COUNT(*)) OVER (), 2) AS pct_of_encounters,
    ROUND(AVG(readmit_30) * 100, 2) AS readmit_rate_pct
FROM vw_eligible_encounters
GROUP BY lace_tier
ORDER BY readmit_rate_pct;


-- name: q17_charlson_comorbidity
-- question: comorbidity burden (charlson index) vs readmission
WITH c AS (
    SELECT LEAST(charlson_index, 4) AS bucket, readmit_30
    FROM vw_eligible_encounters
)
SELECT
    CASE WHEN bucket = 4 THEN '4+' ELSE CAST(bucket AS CHAR) END AS charlson_index,
    COUNT(*)                        AS encounters,
    ROUND(AVG(readmit_30) * 100, 2) AS readmit_rate_pct
FROM c
GROUP BY bucket
ORDER BY bucket;


-- name: q18_equity_check
-- question: readmission by race and gender (groups with 500+ stays)
SELECT
    race,
    gender,
    COUNT(*)                        AS encounters,
    ROUND(AVG(readmit_30) * 100, 2) AS readmit_rate_pct
FROM vw_eligible_encounters
GROUP BY race, gender
HAVING COUNT(*) >= 500
ORDER BY race, gender;


-- name: q19_drug_class_usage
-- question: how common is each diabetes drug class and how do its users do?
WITH enc_class AS (
    SELECT DISTINCT em.encounter_id, md.drug_class
    FROM encounter_medications em
    JOIN medications md ON md.medication_id = em.medication_id
)
SELECT
    ec.drug_class,
    COUNT(*)                                                               AS encounters_on_class,
    ROUND(COUNT(*) * 100 / (SELECT COUNT(*) FROM vw_eligible_encounters), 2) AS pct_of_encounters,
    ROUND(AVG(v.readmit_30) * 100, 2)                                      AS readmit_rate_pct
FROM enc_class ec
JOIN vw_eligible_encounters v ON v.encounter_id = ec.encounter_id
GROUP BY ec.drug_class
ORDER BY encounters_on_class DESC;


-- name: q20_rule_based_cohort
-- question: how good is a simple rule (2+ prior stays AND LACE >= 10) at finding readmissions?
CALL sp_high_risk_cohort(2, 10);
