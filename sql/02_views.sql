-- reusable views so the analysis queries stay short

-- every stay where the patient could actually be readmitted
-- (expired / hospice discharges removed, same as Strack et al. 2014)
CREATE OR REPLACE VIEW vw_eligible_encounters AS
SELECT
    e.*,
    p.race,
    p.gender,
    at.type_group        AS admission_type,
    src.source_group     AS admission_source,
    dd.disposition_group AS discharge_group
FROM encounters e
JOIN patients p               ON p.patient_nbr = e.patient_nbr
JOIN admission_type at        ON at.admission_type_id = e.admission_type_id
JOIN admission_source src     ON src.admission_source_id = e.admission_source_id
JOIN discharge_disposition dd ON dd.discharge_disposition_id = e.discharge_disposition_id
WHERE dd.is_expired_or_hospice = 0;


-- first stay per patient, used for statistical tests (keeps rows independent)
CREATE OR REPLACE VIEW vw_first_encounters AS
SELECT *
FROM (
    SELECT
        v.*,
        ROW_NUMBER() OVER (PARTITION BY v.patient_nbr ORDER BY v.encounter_id) AS visit_seq
    FROM vw_eligible_encounters v
) t
WHERE t.visit_seq = 1;


-- flat feature table for python (diagnoses pivoted, medications aggregated)
CREATE OR REPLACE VIEW vw_model_dataset AS
SELECT
    v.encounter_id,
    v.patient_nbr,
    v.race,
    v.gender,
    v.age_group,
    v.age_mid,
    v.admission_type,
    v.admission_source,
    v.discharge_group,
    v.time_in_hospital,
    v.payer_code,
    v.medical_specialty,
    v.num_lab_procedures,
    v.num_procedures,
    v.num_medications,
    v.number_outpatient,
    v.number_emergency,
    v.number_inpatient,
    v.number_diagnoses,
    -- earlier stays of the same patient in this network (encounter_id order = time order)
    ROW_NUMBER() OVER (PARTITION BY v.patient_nbr ORDER BY v.encounter_id) - 1 AS prior_stays_in_network,
    v.max_glu_serum,
    v.a1c_result,
    v.med_change,
    v.diabetes_med,
    v.charlson_index,
    v.lace_score,
    COALESCE(d.diag1_group, 'Missing') AS diag1_group,
    COALESCE(d.diag2_group, 'Missing') AS diag2_group,
    COALESCE(d.diag3_group, 'Missing') AS diag3_group,
    COALESCE(m.n_active_meds, 0)       AS n_active_meds,
    COALESCE(m.n_dose_changes, 0)      AS n_dose_changes,
    COALESCE(m.insulin_status, 'No')   AS insulin_status,
    COALESCE(m.metformin_status, 'No') AS metformin_status,
    v.readmit_30
FROM vw_eligible_encounters v
LEFT JOIN (
    SELECT
        encounter_id,
        MAX(CASE WHEN diag_position = 1 THEN diag_group END) AS diag1_group,
        MAX(CASE WHEN diag_position = 2 THEN diag_group END) AS diag2_group,
        MAX(CASE WHEN diag_position = 3 THEN diag_group END) AS diag3_group
    FROM encounter_diagnoses
    GROUP BY encounter_id
) d ON d.encounter_id = v.encounter_id
LEFT JOIN (
    SELECT
        em.encounter_id,
        COUNT(*)                                                  AS n_active_meds,
        SUM(em.dosage_status IN ('Up', 'Down'))                   AS n_dose_changes,
        MAX(CASE WHEN md.medication_name = 'insulin'   THEN em.dosage_status END) AS insulin_status,
        MAX(CASE WHEN md.medication_name = 'metformin' THEN em.dosage_status END) AS metformin_status
    FROM encounter_medications em
    JOIN medications md ON md.medication_id = em.medication_id
    GROUP BY em.encounter_id
) m ON m.encounter_id = v.encounter_id;


-- one line headline numbers
CREATE OR REPLACE VIEW vw_kpi_summary AS
SELECT
    COUNT(*)                                  AS total_encounters,
    COUNT(DISTINCT patient_nbr)               AS total_patients,
    SUM(readmit_30)                           AS readmissions_30d,
    ROUND(AVG(readmit_30) * 100, 2)           AS readmit_rate_pct,
    ROUND(AVG(time_in_hospital), 2)           AS avg_length_of_stay,
    SUM(time_in_hospital)                     AS total_bed_days,
    ROUND(AVG(a1c_result <> 'Not Tested') * 100, 2) AS hba1c_tested_pct
FROM vw_eligible_encounters;


-- daily list for the care transition team: highest risk stays + plain english reasons
CREATE OR REPLACE VIEW vw_care_team_worklist AS
SELECT
    rs.encounter_id,
    e.patient_nbr,
    e.age_group,
    dd.disposition_group AS discharge_group,
    e.lace_score,
    ROUND(rs.risk_score * 100, 1) AS predicted_risk_pct,
    rs.risk_percentile,
    rs.risk_tier,
    CONCAT_WS('; ',
        IF(e.number_inpatient >= 2, CONCAT(e.number_inpatient, ' inpatient stays last year'), NULL),
        IF(e.number_emergency >= 1, CONCAT(e.number_emergency, ' ER visits last year'), NULL),
        IF(dd.disposition_group IN ('SNF / Nursing', 'Rehab / LTC', 'Other Facility', 'Other Hospital'),
           CONCAT('discharged to ', dd.disposition_group), NULL),
        IF(e.a1c_result = 'Not Tested', 'no HbA1c test this stay', NULL),
        IF(e.lace_score >= 10, CONCAT('LACE score ', e.lace_score), NULL),
        IF(e.time_in_hospital >= 7, CONCAT(e.time_in_hospital, '-day stay'), NULL)
    ) AS risk_flags
FROM risk_scores rs
JOIN encounters e             ON e.encounter_id = rs.encounter_id
JOIN discharge_disposition dd ON dd.discharge_disposition_id = e.discharge_disposition_id
WHERE rs.risk_tier IN ('High', 'Elevated');
