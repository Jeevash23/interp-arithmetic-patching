from value_transfer.tasks import make_split, predictions, parse, op

def test_generator_oracle_and_controls():
    for split in ['development','validation']:
        data=make_split(split)
        assert data==make_split(split)
        assert len(data)==100
        for row in data:
            d,r=row['donor'],row['recipient']; p=row['predictions']; c=row['controls']
            assert d['first']!=r['first']
            assert len({p[k] for k in ['counterfactual','recipient_answer','donor_answer','donor_z','recipient_z','operand_transfer']})==6
            for t in [d,r,*c.values()]:
                assert op(t['a'],t['b'],t['first'])==t['z']
                assert op(t['z'],t['k'],t['remaining'])==t['y']
            assert c['same_z_decomposition']['a']!=d['a']
            assert c['same_z_decomposition']['z']==d['z']
            assert c['same_z_different_answer']['z']==d['z']
            assert c['same_z_different_answer']['y']!=d['y']
            assert c['different_z_same_answer']['z']!=d['z']
            assert c['different_z_same_answer']['y']==d['y']
            assert predictions(d,r)==p

def test_parser_rejects_incidental_ambiguous_and_partial_answers():
    assert parse('Intermediate: 7\nFinal: 10<|im_end|>')['answer']==10
    assert parse('Intermediate: 7\nI think 10')['answer'] is None
    assert parse('Final: 10\nFinal: 12')['answer'] is None
    assert parse('Final: 10.5')['answer'] is None
    assert parse('Final: -10')['answer']==-10
    assert parse('<think>Final: 10')['answer'] is None
    assert parse('<think>Final: 10</think>\nFinal: 12')['answer']==12
    assert parse('**Final: 12**')['answer']==12

def test_disjoint_template_and_value_splits():
    d,v=make_split('development'),make_split('validation')
    assert not ({r['family'] for r in d}&{r['family'] for r in v})
    assert not ({r['donor']['z'] for r in d}&{r['donor']['z'] for r in v})
