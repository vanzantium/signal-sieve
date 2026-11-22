"""Negative controls for the private nuance candidate. Run directly."""
import unittest
from app import app
from signal_compare import _era_suggestions, _speed_ratio_checks
from signal_sieve import _footer_sponsorship_disclosure as disclosure

TITLE='Source: Journal — "Neon is three times faster than Carbon"'
BODY='Neon is three times faster than Carbon.'
TIMES=('Gemini 4 Neon Extra High: 61, $6.44 per task, 21.5 minutes. '
       'Gemini 4 Neon High: 60, $3.74 per task, 14.8 minutes. '
       'Gemini 4 Carbon High: 60, $7.76 per task, 37.1 minutes.')

class Nuance(unittest.TestCase):
    def post(self,route,body):
        res=app.test_client().post('/api/v1/'+route,json=body)
        self.assertEqual(res.status_code,200,res.get_data(as_text=True))
        return res.get_json()
    def compare(self,*texts):
        return self.post('compare',{'documents':[{'source_name':f'Journal {i}', 'text':t} for i,t in enumerate(texts)]})
    def test_sources_and_tiers(self):
        row=self.compare(TITLE+'\n'+BODY,BODY,TIMES)['speed_ratio_checks'][0]
        self.assertCountEqual([(m['document_index'],m['location']) for m in row['mentions']],[(0,'headline'),(0,'body'),(1,'body')])
        self.assertEqual(row['verdict'],'within_tolerance')
        high,other=row['pairings']
        self.assertEqual(high['within_tolerance'],{'10':False,'15':False,'25':True,'30':True})
        self.assertEqual(other['comparison_scope'],'cross_effort')
        self.assertEqual(high['baseline_samples'][0]['source'],'Journal 2')
    def test_stance(self):
        for text in ['Neon is not three times faster than Carbon.', 'Neon could be three times faster than Carbon.', 'Is Neon three times faster than Carbon?']:
            with self.subTest(text=text):
                row=self.compare(text,TIMES)['speed_ratio_checks'][0]
                self.assertEqual(row['verdict'],'uncheckable')
                self.assertFalse(row['pairings'])
    def test_diagnostic_contract(self):
        text=TITLE+'\nCoverage follows the official announcement today.'
        a=self.post('analyze',{'text':text,'source_name':'Journal 0'})
        self.assertEqual(a['headline_weighting'][0], self.compare(text,TIMES)['headline_weighting'][0])
        no=self.post('analyze',{'text':'Coverage follows the official announcement today.'})
        self.assertNotIn('headline_weighting',no)
    def test_metadata_never_enters_facts(self):
        t='Source: Journal — "An update. GPQA: 94%. Input costs $217 per million tokens."\nThe report discusses testing.'
        result=self.compare(t,t)
        for field in ('alignment','conflicts','unique_claims','shared_claims'):
            self.assertFalse(result[field],field)
    def test_short_footer(self):
        for text in ['The work was funded by Delta Fund.', 'A short note.\nThe work was funded by Delta Fund.',
                     'Their motto reads:\nSponsored by Acme was printed on every shirt.']:
            self.assertFalse(disclosure(text)['present'],text)
        for text in ['Reporting ends.\nFunded by U.S. Steel Corp.', 'Disclosure: Funded by U.S. Steel Corp.']:
            self.assertEqual(disclosure(text)['org'],'U.S. Steel Corp')
    def test_adjacent_era(self):
        price='Carbon costs $7.76 per task.'
        context='That Carbon rate takes effect on December 1.'
        hints=_era_suggestions(price+' '+context,'Article',2)
        self.assertEqual(len(hints),1)
        self.assertEqual(hints[0]['era'],'unknown')
        self.assertEqual((hints[0]['price_span'],hints[0]['context_span']),(price,context))
        for t in [price+' '+context.replace('Carbon','Neon'),price+' That rate takes effect on December 1.',
                  price+' Another example follows. '+context,price+'\n\n'+context,
                  price+' '+context.replace('takes','might take'),price+' '+context.replace('1.','1 or December 2.'),
                  'If '+price+' '+context,price.replace('$7.76','$7.76 or $8')+' '+context]:
            self.assertFalse(_era_suggestions(t,'Article',2),t)
        self.assertFalse(self.compare(price,context)['era_suggestions'])
    def test_ambiguous_measurements_and_unknown_tiers(self):
        claim={'subject':'s','numeric_value':3,'entity_key':'mult','multiplier_baseline':'b'}
        def duration(model,val,eff):
            return dict(metric='duration',subject=model,numeric_value=val,unit='minutes',effort=eff)
        for eff,expected in [('high','insufficient_or_ambiguous_measurements'),('','no_same_effort_pair')]:
            row=_speed_ratio_checks([{'claim':claim,'doc_idx':0}],[[duration('s',1,eff),duration('s',2,eff),duration('b',3,eff)]],str)[0]
            self.assertEqual(row['verdict'],expected)
if __name__=='__main__':
    unittest.main(verbosity=2)
