-- normalized schema for the diabetes readmission data
-- 4 lookup tables + patients + encounters + 2 bridge tables + model scores

SET FOREIGN_KEY_CHECKS = 0;
DROP TABLE IF EXISTS risk_scores;
DROP TABLE IF EXISTS encounter_medications;
DROP TABLE IF EXISTS encounter_diagnoses;
DROP TABLE IF EXISTS encounters;
DROP TABLE IF EXISTS patients;
DROP TABLE IF EXISTS medications;
DROP TABLE IF EXISTS admission_type;
DROP TABLE IF EXISTS admission_source;
DROP TABLE IF EXISTS discharge_disposition;
SET FOREIGN_KEY_CHECKS = 1;

-- lookup tables (from IDS_mapping.csv + my own grouping)
CREATE TABLE admission_type (
    admission_type_id   TINYINT UNSIGNED PRIMARY KEY,
    description         VARCHAR(60)  NOT NULL,
    type_group          VARCHAR(20)  NOT NULL
);

CREATE TABLE admission_source (
    admission_source_id TINYINT UNSIGNED PRIMARY KEY,
    description         VARCHAR(80)  NOT NULL,
    source_group        VARCHAR(20)  NOT NULL
);

CREATE TABLE discharge_disposition (
    discharge_disposition_id TINYINT UNSIGNED PRIMARY KEY,
    description              VARCHAR(120) NOT NULL,
    disposition_group        VARCHAR(30)  NOT NULL,
    is_expired_or_hospice    TINYINT(1)   NOT NULL DEFAULT 0
);

CREATE TABLE medications (
    medication_id   TINYINT UNSIGNED PRIMARY KEY,
    medication_name VARCHAR(40) NOT NULL UNIQUE,
    drug_class      VARCHAR(40) NOT NULL
);

-- one row per patient
CREATE TABLE patients (
    patient_nbr INT UNSIGNED PRIMARY KEY,
    race        VARCHAR(20) NOT NULL,
    gender      VARCHAR(10) NOT NULL
);

-- one row per hospital stay (fact table)
CREATE TABLE encounters (
    encounter_id             INT UNSIGNED PRIMARY KEY,
    patient_nbr              INT UNSIGNED NOT NULL,
    admission_type_id        TINYINT UNSIGNED NOT NULL,
    discharge_disposition_id TINYINT UNSIGNED NOT NULL,
    admission_source_id      TINYINT UNSIGNED NOT NULL,
    age_group                VARCHAR(10) NOT NULL,
    age_mid                  TINYINT UNSIGNED NOT NULL,
    time_in_hospital         TINYINT UNSIGNED NOT NULL,
    payer_code               VARCHAR(10) NOT NULL,
    medical_specialty        VARCHAR(60) NOT NULL,
    num_lab_procedures       SMALLINT UNSIGNED NOT NULL,
    num_procedures           TINYINT UNSIGNED NOT NULL,
    num_medications          SMALLINT UNSIGNED NOT NULL,
    number_outpatient        SMALLINT UNSIGNED NOT NULL,
    number_emergency         SMALLINT UNSIGNED NOT NULL,
    number_inpatient         SMALLINT UNSIGNED NOT NULL,
    number_diagnoses         TINYINT UNSIGNED NOT NULL,
    max_glu_serum            VARCHAR(12) NOT NULL,
    a1c_result               VARCHAR(12) NOT NULL,
    med_change               TINYINT(1) NOT NULL,
    diabetes_med             TINYINT(1) NOT NULL,
    charlson_index           TINYINT UNSIGNED NOT NULL,
    lace_score               TINYINT UNSIGNED NOT NULL,
    readmitted               VARCHAR(5) NOT NULL,
    readmit_30               TINYINT(1) NOT NULL,
    CONSTRAINT fk_enc_patient   FOREIGN KEY (patient_nbr) REFERENCES patients (patient_nbr),
    CONSTRAINT fk_enc_adm_type  FOREIGN KEY (admission_type_id) REFERENCES admission_type (admission_type_id),
    CONSTRAINT fk_enc_discharge FOREIGN KEY (discharge_disposition_id) REFERENCES discharge_disposition (discharge_disposition_id),
    CONSTRAINT fk_enc_source    FOREIGN KEY (admission_source_id) REFERENCES admission_source (admission_source_id),
    CONSTRAINT chk_los CHECK (time_in_hospital BETWEEN 1 AND 14),
    CONSTRAINT chk_readmit CHECK (readmit_30 IN (0, 1))
);

-- up to 3 diagnoses per encounter (diag_1 = primary)
CREATE TABLE encounter_diagnoses (
    encounter_id  INT UNSIGNED NOT NULL,
    diag_position TINYINT UNSIGNED NOT NULL,
    icd9_code     VARCHAR(10) NOT NULL,
    diag_group    VARCHAR(20) NOT NULL,
    PRIMARY KEY (encounter_id, diag_position),
    CONSTRAINT fk_diag_enc FOREIGN KEY (encounter_id) REFERENCES encounters (encounter_id)
);

-- only the diabetes drugs actually given (the raw file has 23 mostly "No" columns)
CREATE TABLE encounter_medications (
    encounter_id  INT UNSIGNED NOT NULL,
    medication_id TINYINT UNSIGNED NOT NULL,
    dosage_status ENUM('Steady', 'Up', 'Down') NOT NULL,
    PRIMARY KEY (encounter_id, medication_id),
    CONSTRAINT fk_med_enc  FOREIGN KEY (encounter_id) REFERENCES encounters (encounter_id),
    CONSTRAINT fk_med_drug FOREIGN KEY (medication_id) REFERENCES medications (medication_id)
);

-- filled by src/risk_model.py (out-of-fold model scores, so no stay is scored by a model that saw it)
CREATE TABLE risk_scores (
    encounter_id    INT UNSIGNED PRIMARY KEY,
    risk_score      DECIMAL(6, 4) NOT NULL,
    risk_percentile TINYINT UNSIGNED NOT NULL,
    risk_tier       VARCHAR(10) NOT NULL,
    model_version   VARCHAR(40) NOT NULL,
    scored_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_score_enc FOREIGN KEY (encounter_id) REFERENCES encounters (encounter_id)
);

-- indexes for the columns I filter / group on the most
CREATE INDEX idx_enc_patient   ON encounters (patient_nbr);
CREATE INDEX idx_enc_discharge ON encounters (discharge_disposition_id);
CREATE INDEX idx_enc_readmit   ON encounters (readmit_30);
CREATE INDEX idx_enc_age       ON encounters (age_group);
CREATE INDEX idx_diag_group    ON encounter_diagnoses (diag_group, diag_position);
CREATE INDEX idx_med_drug      ON encounter_medications (medication_id, dosage_status);
CREATE INDEX idx_score_tier    ON risk_scores (risk_tier);
