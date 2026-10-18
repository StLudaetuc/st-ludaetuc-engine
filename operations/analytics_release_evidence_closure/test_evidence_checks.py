import unittest
from operations.analytics_release_evidence_closure.evidence_checks import artifact_errors,publication_errors,summarise_access
class Checks(unittest.TestCase):
 def sample(self,**kw):
  r=dict(record_id='a',at='2026-10-16T12:00:01Z',route='/analytics/export',status=401);r.update(kw);return r
 def summary(self,rows,**kw):return summarise_access(rows,'2026-10-16T12:00:00Z','2026-10-16T12:05:00Z',**kw)
 def test_empty_not_verified_zero(self):self.assertEqual(self.summary([])['coverage'],'UNKNOWN_OR_PARTIAL')
 def test_pagination_duplicate_id(self):self.assertEqual(self.summary([self.sample(),self.sample()])['observed_responses'],1)
 def test_distinct_identical_requests_count(self):self.assertEqual(self.summary([self.sample(),self.sample(record_id='b')])['observed_responses'],2)
 def test_data_200_incident(self):self.assertEqual(self.summary([self.sample(status=200)])['unexpected_data_2xx'],1)
 def test_status_200_separate(self):self.assertEqual(self.summary([self.sample(status=200,route='/analytics/access-status')])['unexpected_data_2xx'],0)
 def test_end_excluded(self):self.assertEqual(self.summary([self.sample(at='2026-10-16T12:05:00Z')])['observed_responses'],0)
 def test_invalid_time_unknown(self):self.assertEqual(self.summary([self.sample(at='2026-10-16T12:00:01')],complete=True)['coverage'],'UNKNOWN_OR_PARTIAL')
 def test_bad_status_unknown(self):self.assertEqual(self.summary([self.sample(status=True)])['invalid'],1)
 def test_partial_not_full(self):self.assertEqual(self.summary([self.sample()])['coverage'],'UNKNOWN_OR_PARTIAL')
 def test_false_publication_rejected(self):self.assertTrue(publication_errors({'status':'VERIFIED_PUBLISHED'}))
 def test_author_not_independent(self):self.assertTrue(publication_errors({'author_test_evidence':True,'independent_assurance':True}))
 def test_missing_fallback_rejected(self):self.assertIn('fallback files absent',artifact_errors({'fallback_activation':True,'fallback_files_present':False}))
 def test_merge_sha_not_branch_sha(self):self.assertIn('result SHA not approved before ref change',artifact_errors({'merge_result_sha':'new','approved_result_sha':'old'}))
 def test_forbidden_recovery(self):self.assertIn('prohibited bare recovery/release target',artifact_errors({'recovery_sha':'2ca3fa49f1b6e44be55c5ccdc1dbddc44d9f5b3c'}))
 def test_conflicting_log_identity_unknown(self):self.assertEqual(self.summary([self.sample(),self.sample(status=200)],complete=True)['coverage'],'UNKNOWN_OR_PARTIAL')
 def test_good_record_not_authority(self):
  m=dict(merge_result_sha='a'*40,approved_result_sha='a'*40,merge_result_tree='b'*40,approved_result_tree='b'*40,dependencies_reproducible=True,fallback_files_present=True,fallback_activation=True)
  self.assertEqual(artifact_errors(m),[]) # consistency only; not proof of signatures/approval
if __name__=='__main__':unittest.main(verbosity=2)
