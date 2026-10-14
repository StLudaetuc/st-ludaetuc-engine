-- PROPOSED operator evidence acquisition. Never executed by this candidate.
-- Use an approved read-only role, transaction READ ONLY, bounded UTC window.
-- Bind %(start)s/%(end)s/%(limit)s through DB driver, not string substitution.
BEGIN TRANSACTION READ ONLY;
SELECT event_type,count(*),count(DISTINCT event_id),min(event_time),max(event_time),sum(retry_count)
FROM events WHERE event_time >= %(start)s AND event_time < %(end)s GROUP BY event_type;
SELECT event_id,event_type,instrument,timeframe,event_time,received_at,retry_count,
 payload->>'time' AS source_time,payload->>'time_close' AS source_close_time,
 payload->>'ticker' AS source_ticker,payload->>'interval' AS source_interval
FROM events WHERE event_type='market_data' AND event_time >= %(start)s AND event_time < %(end)s
ORDER BY event_time,event_id LIMIT %(limit)s;
SELECT o.event_id,o.trade_id AS broker_trade_id,o.event_time,o.received_at,o.retry_count,
 topen.event_id AS opened_event_id,topen.trade_id AS signal_id,p.event_id AS prediction_event_id,
 topen.payload->>'fill_status' AS fill_status,p.strategy_version,p.model_version
FROM events o LEFT JOIN events topen ON topen.event_type='trade_opened' AND topen.payload->>'oanda_trade_id'=o.trade_id
LEFT JOIN events p ON p.event_type='prediction' AND p.trade_id=topen.trade_id
WHERE o.event_type='trade_outcome' AND o.event_time >= %(start)s AND o.event_time < %(end)s
ORDER BY o.event_time,o.event_id LIMIT %(limit)s;
-- JSON null must not shadow a genuine hypothetical plan. Future reviewed
-- endpoint adaptation must retain source context and explicit hypothetical label.
SELECT event_id,trade_id,event_time,received_at,
 COALESCE(NULLIF(payload->'final_trade_plan','null'::jsonb),payload->'hypothetical_trade_plan') AS research_plan,
 payload->>'rejection_reason_code' AS rejection_reason
FROM events WHERE event_type='prediction' AND event_time >= %(start)s AND event_time < %(end)s
ORDER BY event_time,event_id LIMIT %(limit)s;
ROLLBACK;
