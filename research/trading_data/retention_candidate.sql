-- DRAFT; local isolated research schema only. NOT applied to production.
BEGIN;
CREATE SCHEMA research_candidate;
REVOKE ALL ON SCHEMA research_candidate FROM PUBLIC;
CREATE TABLE research_candidate.facts(
 fact_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 account_alias text NOT NULL,broker_environment text NOT NULL CHECK(broker_environment IN ('OANDA_PRACTICE','OANDA_LIVE')),
 source_event_id text NOT NULL,event_kind text NOT NULL,event_time timestamptz NOT NULL,
 source_digest text NOT NULL CHECK(source_digest ~ '^[0-9a-f]{64}$'),
 raw_evidence_ref text NOT NULL,payload jsonb NOT NULL CHECK(octet_length(payload::text)<=65536),
 source_schema text NOT NULL,server_received_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 UNIQUE(account_alias,broker_environment,event_kind,source_event_id,source_digest)
);
CREATE TABLE research_candidate.deliveries(
 delivery_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,fact_id bigint NOT NULL REFERENCES research_candidate.facts,
 source_received_at timestamptz NOT NULL,server_received_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 classification text NOT NULL CHECK(classification IN ('NEW_FACT','EXACT_REPLAY','CONFLICT','CORRECTION'))
);
CREATE TABLE research_candidate.corrections(
 correction_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,prior_fact_id bigint NOT NULL REFERENCES research_candidate.facts,
 replacement_fact_id bigint NOT NULL REFERENCES research_candidate.facts,review_evidence_ref text NOT NULL,
 CHECK(prior_fact_id<>replacement_fact_id)
);
CREATE FUNCTION research_candidate.same_scope() RETURNS trigger LANGUAGE plpgsql SET search_path=pg_catalog,research_candidate AS $$
BEGIN
 IF NOT EXISTS(SELECT 1 FROM research_candidate.facts a JOIN research_candidate.facts b ON a.account_alias=b.account_alias AND a.broker_environment=b.broker_environment AND a.event_kind=b.event_kind WHERE a.fact_id=NEW.prior_fact_id AND b.fact_id=NEW.replacement_fact_id) THEN RAISE EXCEPTION 'same-scope correction required'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER same_scope BEFORE INSERT ON research_candidate.corrections FOR EACH ROW EXECUTE FUNCTION research_candidate.same_scope();
CREATE FUNCTION research_candidate.immutable() RETURNS trigger LANGUAGE plpgsql SET search_path=pg_catalog AS $$
BEGIN RAISE EXCEPTION 'immutable research evidence'; END $$;
DO $$ DECLARE t text; BEGIN
 FOREACH t IN ARRAY ARRAY['facts','deliveries','corrections'] LOOP
 EXECUTE format('CREATE TRIGGER immutable_rows BEFORE UPDATE OR DELETE ON research_candidate.%I FOR EACH ROW EXECUTE FUNCTION research_candidate.immutable()',t);
 EXECUTE format('CREATE TRIGGER immutable_truncate BEFORE TRUNCATE ON research_candidate.%I FOR EACH STATEMENT EXECUTE FUNCTION research_candidate.immutable()',t);
 END LOOP;
END $$;
REVOKE ALL ON ALL TABLES IN SCHEMA research_candidate FROM PUBLIC;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA research_candidate FROM PUBLIC;
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA research_candidate FROM PUBLIC;
COMMIT;
