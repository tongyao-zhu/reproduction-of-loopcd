import unittest
from sharded_aime_common import combine_unique,expected_keys,common

class ShardTests(unittest.TestCase):
    def test_partition_no_overlap_full_coverage(self):
        prompts=[dict(task_id=str(i)) for i in range(30)]
        keys=expected_keys(prompts)
        self.assertEqual(len(keys),480)
        previous={k:k for k in keys[:79]}
        a={k:k for k in expected_keys(prompts,0,18) if k not in previous}
        b={k:k for k in expected_keys(prompts,18,30)}
        self.assertEqual(combine_unique([previous,a,b],keys),{k:k for k in keys})
    def test_reject_duplicate_even_if_identical(self):
        with self.assertRaises(ValueError):combine_unique([{'x':1},{'x':1}],['x'])
    def test_reject_missing_extra(self):
        for parts in ([{'x':1}],[{'x':1,'y':2,'z':3}]):
            with self.subTest(parts=parts),self.assertRaises(ValueError):combine_unique(parts,['x','y'])
    def test_only_lineage_fields_may_differ(self):
        self.assertEqual(common(dict(engine='fixed',source_commit='a',shard={})),common(dict(engine='fixed',source_commit='b',shard={'start':18})))
        self.assertNotEqual(common(dict(engine='fixed')),common(dict(engine='changed')))

if __name__=='__main__':unittest.main()
