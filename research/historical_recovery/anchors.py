"""Verifier-owned offline expectations. No anchor issuance/approval mechanism.

Caller must obtain this object from an independently retained, authorised channel.
A producer can fabricate an anchor file: matching it is not proof that its custodian
was independent. That custody/governance requirement is outside this pure library.
"""
from dataclasses import dataclass
import re
from .normalizer import iso_time
SHA=re.compile(r'^[0-9a-f]{64}$')

@dataclass(frozen=True)
class TrustedAnchor:
    expected_manifest_hash: str
    expected_audit_sha256: str
    source_count: int
    exported_count: int
    scope: dict
    window: dict
    acquired_at: str
    operator_identity_reference: str
    manifest_version: str
    redaction_version: str
    # Stable C/event-ID ordering: id, unredacted event, redacted bare event, raw payload.
    records: tuple
    custody_reference: str

    def __post_init__(self):
        if not SHA.fullmatch(self.expected_manifest_hash) or not SHA.fullmatch(self.expected_audit_sha256):raise ValueError('INVALID_EXPECTED_ANCHOR')
        if type(self.source_count) is not int or type(self.exported_count) is not int or not 0<=self.exported_count<=self.source_count or self.exported_count>50000:raise ValueError('INVALID_ANCHOR_COUNTS')
        if not isinstance(self.custody_reference,str) or not re.fullmatch(r'[A-Za-z0-9_.:/-]{1,160}',self.custody_reference):raise ValueError('ANCHOR_CUSTODY_REFERENCE_REQUIRED')
        iso_time(self.acquired_at)
        if len(self.records)!=self.exported_count:raise ValueError('ANCHOR_RECORD_COUNT')
        ids=[]
        for record in self.records:
            if len(record)!=4 or any(not isinstance(v,str) or not SHA.fullmatch(v) for v in record):raise ValueError('INVALID_RECORD_ANCHOR')
            ids.append(record[0])
        if ids!=sorted(set(ids)):raise ValueError('ANCHOR_ORDER_OR_DUPLICATES')

def compare_anchor(anchor,manifest,rows,audit_sha256):
    if anchor is None:return {'state':'UNPROVED','reason':'INDEPENDENT_ANCHOR_NOT_SUPPLIED','custody_reference':None}
    if not isinstance(anchor,TrustedAnchor):raise ValueError('INVALID_ANCHOR_TYPE')
    checks=[(manifest['manifest_hash']==anchor.expected_manifest_hash,'EXPECTED_MANIFEST_MISMATCH'),(audit_sha256==anchor.expected_audit_sha256,'EXPECTED_AUDIT_MISMATCH'),(manifest['source_count']==anchor.source_count and manifest['exported_count']==anchor.exported_count,'EXPECTED_COUNTS_MISMATCH'),(manifest['scope']==anchor.scope,'EXPECTED_SCOPE_MISMATCH'),(manifest['window']==anchor.window,'EXPECTED_WINDOW_MISMATCH'),(manifest['acquired_at']==anchor.acquired_at,'EXPECTED_ACQUISITION_MISMATCH'),(manifest['operator_identity_reference']==anchor.operator_identity_reference,'EXPECTED_OPERATOR_MISMATCH'),(manifest['contract']==anchor.manifest_version and manifest['redaction_policy']==anchor.redaction_version,'EXPECTED_VERSION_MISMATCH')]
    actual=tuple((r['event_id'],r['acquisition_source_event_digest'],r['acquisition_redacted_event_digest'],r['acquisition_source_payload_digest']) for r in rows)
    checks.append((actual==anchor.records,'EXPECTED_RECORD_POPULATION_MISMATCH'))
    for matches,reason in checks:
        if not matches:raise ValueError(reason)
    return {'state':'MATCHED_INDEPENDENT_EXPECTATION','reason':'ALL_EXPECTED_IDENTITIES_DIGESTS_COUNTS_MATCH','custody_reference':anchor.custody_reference,'expected_manifest_hash':anchor.expected_manifest_hash,'expected_audit_sha256':anchor.expected_audit_sha256}
