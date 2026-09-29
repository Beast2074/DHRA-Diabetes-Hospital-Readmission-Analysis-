-- stored function + procedures the care team could reuse

DROP FUNCTION IF EXISTS fn_lace_risk_tier;
DROP PROCEDURE IF EXISTS sp_readmission_profile;
DROP PROCEDURE IF EXISTS sp_high_risk_cohort;

DELIMITER $$

-- standard LACE cut offs: 0-4 low, 5-9 moderate, 10+ high
CREATE FUNCTION fn_lace_risk_tier(p_score INT)
RETURNS VARCHAR(10)
DETERMINISTIC NO SQL
BEGIN
    RETURN CASE
        WHEN p_score >= 10 THEN 'High'
        WHEN p_score >= 5  THEN 'Moderate'
        ELSE 'Low'
    END;
END $$


-- readmission rate broken down by any one column, e.g. CALL sp_readmission_profile('age_group');
-- column name is checked against a whitelist so it can't be abused for sql injection
CREATE PROCEDURE sp_readmission_profile(IN p_dimension VARCHAR(40))
BEGIN
    IF p_dimension NOT IN ('age_group', 'race', 'gender', 'admission_type', 'admission_source',
                           'discharge_group', 'a1c_result', 'max_glu_serum', 'payer_code',
                           'medical_specialty', 'time_in_hospital', 'number_inpatient') THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'dimension not allowed';
    END IF;

    SET @q = CONCAT(
        'SELECT ', p_dimension, ' AS segment, COUNT(*) AS encounters, SUM(readmit_30) AS readmissions, ',
        'ROUND(AVG(readmit_30) * 100, 2) AS readmit_rate_pct ',
        'FROM vw_eligible_encounters GROUP BY ', p_dimension,
        ' HAVING COUNT(*) >= 100 ORDER BY readmit_rate_pct DESC'
    );
    PREPARE stmt FROM @q;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
END $$


-- compares a rule based cohort (prior admissions + LACE) against everyone else
CREATE PROCEDURE sp_high_risk_cohort(IN p_min_prior_inpatient INT, IN p_min_lace INT)
BEGIN
    SELECT
        CASE WHEN number_inpatient >= p_min_prior_inpatient AND lace_score >= p_min_lace
             THEN 'Flagged cohort' ELSE 'Everyone else' END AS cohort,
        COUNT(*)                                     AS encounters,
        ROUND(COUNT(*) * 100 / SUM(COUNT(*)) OVER (), 2) AS pct_of_encounters,
        SUM(readmit_30)                              AS readmissions,
        ROUND(SUM(readmit_30) * 100 / SUM(SUM(readmit_30)) OVER (), 2) AS pct_of_readmissions,
        ROUND(AVG(readmit_30) * 100, 2)              AS readmit_rate_pct
    FROM vw_eligible_encounters
    GROUP BY cohort
    ORDER BY cohort DESC;
END $$

DELIMITER ;
