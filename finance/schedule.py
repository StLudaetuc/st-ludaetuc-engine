"""Versioned deterministic EOD adapter. CEO-reported intent is not Make proof.

Calendar deadlines and completed evidence watermarks are distinct. This module
does no broker/Make/network calls and never assumes an empty successful batch.
"""
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from .policy import instant, evidence_ref

VERSION = 'london_eod_tues_sat_0030_v1'

@dataclass(frozen=True)
class Schedule:
    version: str
    timezone: str
    weekdays: tuple
    hour: int
    minute: int
    grace_seconds: int
    configuration_evidence: str | None
    configuration_verified: bool
    def __post_init__(self):
        if (self.version,self.timezone,self.weekdays,self.hour,self.minute)!=(VERSION,'Europe/London',(1,2,3,4,5),0,30):
            raise ValueError('unreviewed calendar version')
        if type(self.grace_seconds) is not int or self.grace_seconds<0 or self.grace_seconds>86400:
            raise ValueError('explicit bounded completion grace required')
        if type(self.configuration_verified) is not bool:
            raise ValueError('explicit configuration verification required')
        if self.configuration_evidence is not None: evidence_ref(self.configuration_evidence)
        if self.configuration_verified and self.configuration_evidence is None:
            raise ValueError('configuration verification needs evidence')

def next_sync_due(closed_at, schedule):
    if not isinstance(schedule,Schedule): raise ValueError('explicit versioned schedule required')
    event=instant(closed_at);zone=ZoneInfo(schedule.timezone)
    # A run covers events strictly before its scheduled instant, not a guessed
    # broker calendar date. Query adapters must retain a complete catch-up cursor.
    day=event.astimezone(zone).date()
    for offset in range(9):
        d=day+timedelta(days=offset)
        candidate=datetime.combine(d,time(schedule.hour,schedule.minute),zone)
        if d.weekday() in schedule.weekdays and instant(candidate)>event:
            return instant(candidate)
    raise ValueError('no schedule boundary')

@dataclass(frozen=True)
class SyncRun:
    run_id: str
    scheduled_at: object
    started_at: object
    completed_at: object | None
    state: str
    watermark_start: object
    watermark_end: object
    complete: bool
    evidence_count: int
    reviewed_empty: bool
    manifest_ref: str
    def __post_init__(self):
        if not self.run_id or self.state not in {'COMPLETED','FAILED','PARTIAL','STARTED'}: raise ValueError('invalid run identity/state')
        for key in ('scheduled_at','started_at','watermark_start','watermark_end'):
            object.__setattr__(self,key,instant(getattr(self,key)))
        if self.completed_at is not None: object.__setattr__(self,'completed_at',instant(self.completed_at))
        if self.watermark_end<=self.watermark_start or self.started_at<self.scheduled_at: raise ValueError('invalid sync range')
        if self.completed_at is not None and self.completed_at<self.started_at: raise ValueError('invalid receipt order')
        if type(self.complete) is not bool or type(self.reviewed_empty) is not bool or type(self.evidence_count) is not int or self.evidence_count<0: raise ValueError('explicit sync population required')
        if self.complete and (self.state!='COMPLETED' or self.completed_at is None or (self.evidence_count==0)!=self.reviewed_empty): raise ValueError('complete/empty evidence mismatch')
        evidence_ref(self.manifest_ref)

def sync_status(event_at, as_of, schedule, runs):
    event,now=instant(event_at),instant(as_of);due=next_sync_due(event,schedule)
    if not schedule.configuration_verified:
        return {'state':'INVESTIGATION_REQUIRED','reason':'schedule_configuration_unverified','due_at':due,'verified':False}
    unique={}
    for r in runs:
        if not isinstance(r,SyncRun): raise ValueError('verified run evidence required')
        if r.run_id in unique and unique[r.run_id]!=r:
            return {'state':'INVESTIGATION_REQUIRED','reason':'conflicting_sync_run','due_at':due,'verified':True}
        unique[r.run_id]=r
    complete=[r for r in unique.values() if r.complete and r.completed_at<=now and r.watermark_start<=event<r.watermark_end]
    if complete:
        return {'state':'MATCHED','reason':'complete_catch_up_watermark','due_at':due,'verified':True}
    deadline=due+timedelta(seconds=schedule.grace_seconds)
    state='PENDING_EOD_SYNC' if now<deadline else 'INVESTIGATION_REQUIRED'
    reason='awaiting_scheduled_sync' if now<deadline else 'missed_or_incomplete_sync_run'
    return {'state':state,'reason':reason,'due_at':deadline,'verified':True}
