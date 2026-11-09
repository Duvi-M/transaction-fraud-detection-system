-- Registered input: transactions. Time bands are elapsed hours, not local clock hours.
SELECT COUNT(*) AS transactions, SUM(Class) AS frauds, AVG(Class) AS fraud_fraction,
       AVG(Amount) AS mean_amount FROM transactions;
SELECT FLOOR(Time / 3600) AS elapsed_hour, COUNT(*) AS transactions,
       SUM(Class) AS frauds, AVG(Class) AS fraud_fraction
FROM transactions GROUP BY 1 ORDER BY 1;
SELECT CASE WHEN Amount < 10 THEN '0-10' WHEN Amount < 100 THEN '10-100'
            WHEN Amount < 1000 THEN '100-1000' ELSE '1000+' END AS amount_band,
       COUNT(*) AS transactions, SUM(Class) AS frauds, AVG(Class) AS fraud_fraction
FROM transactions GROUP BY 1 ORDER BY 1;
SELECT Class, COUNT(*) AS transactions, MEDIAN(Amount) AS median_amount,
       QUANTILE_CONT(Amount, 0.95) AS p95_amount FROM transactions GROUP BY Class;
