-- UNEXECUTED template: authorised SELECT-only operator, approved internal host.
-- psql -X -v ON_ERROR_STOP=1 -v start_utc='APPROVED_START' -v end_utc='APPROVED_END'
-- Existing approved connection selected privately; no connection string in evidence.
BEGIN READ ONLY;
SET LOCAL statement_timeout = '5s';
-- Expected source schema must be checked separately; never repair missing schema here.
SELECT event_type, status, count(*) AS current_rows,
       min(received_at) AS first_received_at, max(received_at) AS last_received_at
FROM events
WHERE received_at >= :'start_utc'::timestamptz
  AND received_at < :'end_utc'::timestamptz
  AND :'end_utc'::timestamptz > :'start_utc'::timestamptz
  AND :'end_utc'::timestamptz - :'start_utc'::timestamptz <= interval '1 hour'
  AND event_type IN ('prediction','market_data','trade_opened','trade_outcome')
GROUP BY event_type, status
ORDER BY event_type, status
LIMIT 100;
ROLLBACK;
-- Empty output (including invalid/out-of-bounds window) is UNPROVED; never verified zero.
-- SQL syntax/privileges/plan/runtime not validated against a database by this task.
