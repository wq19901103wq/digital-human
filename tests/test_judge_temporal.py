"""Temporal familiar-contact isolation, fresh paired draws, and safe resume."""
import copy
import json
from collections import Counter

import pytest

from scripts.legacy import prepare_judge_temporal as split
from scripts.legacy import retrain_judge_lr_temporal as job
from scripts.legacy import verify_judge_temporal as audit
from src import cache, tracing
from src.config import ConfigError, sha256_file
from src.generator.generator import ReplyGenerator
from src.iteration import experiment, runner, versions
from src.judge import lr_retrain as lr, corrected, corrected_v1 as rt
from test_iteration import priv, _write
from test_lr_retrain import FeatureClient
from test_corrected_judge import bundle, case, features


def messages(contacts=2, sessions=16):
    rows=[]
    for kind in ('group','private'):
        for contact in range(contacts):
            chat=f'{kind}:account:{contact}'
            for session in range(sessions):
                for offset,own in enumerate((False,True,True)):
                    rows.append({'chat_id':chat,'source_chat_id':chat,'chat_type':kind,'chat_name':chat,
                        'timestamp':1700000000+session*40000+(10000 if offset==2 else offset),
                        'is_self':own,'sender':'self' if own else 'other','text':f'{session}/{offset}'})
    return rows


def test_complete_bursts_and_original_offsets_with_familiar_contacts():
    source=messages()
    parts,cuts,purged=split.partition_examples(source)
    # The existing context window can include earlier sessions. Those windows
    # must be purged when they cross a split, while whole self bursts survive.
    originals=split.extract_examples(source)
    retained={r['id'] for rows in parts.values() for r in rows}
    omitted=[r for r in originals if r['id'] not in retained]
    assert len(omitted)==purged['crosses_boundary']>0
    for r in omitted:
        chat,_,offset=r['source_message_id'].rpartition(':')
        start,end=int(offset)-len(r['context_messages']),int(offset)+len(r['reply'])
        assert any(start<cuts[chat][key]<end for key in ('train_end_index','development_end_index'))
    contacts=[]
    for part,rows in parts.items():
        contacts.append({r['source_span']['chat_id'] for r in rows})
        for r in rows:
            span=r['source_span'];chat=span['chat_id'];a=cuts[chat]['train_end_index'];b=cuts[chat]['development_end_index']
            assert len(r['reply'])==2  # even across a >2 hour gap between self bubbles
            assert span['end']-span['reply_start']==2
            assert int(r['source_message_id'].rpartition(':')[2])==span['reply_start']
            low,high={'train':(0,a),'development':(a,b),'fixed_test':(b,48)}[part]
            assert low<=span['start']<span['end']<=high
    assert contacts[0]==contacts[1]==contacts[2]
    selected=split.choose(parts['train'],{'group':3,'private':3},2,42,contacts[1])
    assert contacts[1]<={r['source_span']['chat_id'] for r in selected}
    assert max(Counter(r['source_span']['chat_id'] for r in selected).values())<=2


def test_temporal_reference_filter_and_failure_cannot_silently_fallback(monkeypatch):
    target={'source_span':{'chat_id':'same','start':10,'start_timestamp':100}}
    def row(chat,end,time,part='train'):
        return {'source_span':{'chat_id':chat,'end':end,'end_timestamp':time},'temporal_partition':part}
    pool={'past':row('same',5,50),'overlap':row('same',11,50),'future':row('other',1,100),
        'evaluation':row('other',1,50,'development'),'other-past':row('other',50,99)}
    excluded=split.TemporalExclusions(pool,target,{'manual'})
    assert not any(k in excluded for k in ('past','other-past'))
    assert all(k in excluded for k in ('overlap','future','evaluation','unknown','manual'))
    class BrokenRetriever:
        def retrieve(self,**kw):return [{'id':'future'}]
    wrapped=split.TemporalRetriever(BrokenRetriever(),pool);wrapped.case=target
    with pytest.raises(ConfigError,match='未来'):
        wrapped.retrieve()
    fake=object.__new__(job.common.TrainingGenerator);fake._retriever=wrapped
    monkeypatch.setattr(ReplyGenerator,'build_prompt',lambda *a,**k:[])
    with pytest.raises(ConfigError,match='禁止降级'):
        fake.build_prompt(target)


def test_source_audit_checks_formal_selection_against_raw_spans(priv,monkeypatch,tmp_path):
    source=messages(contacts=40,sessions=120)
    parts,cuts,_=split.partition_examples(source)
    selected={p:split.choose(parts[p],{'group':750,'private':250},25,42) for p in ('development','fixed_test')}
    required={r['source_span']['chat_id'] for rows in selected.values() for r in rows}
    selected['train']=split.choose(parts['train'],{'group':600,'private':600},125,42,required)
    data=priv/'data/d-0001'
    for part,name in [('train','train_pool.jsonl'),('development','dev_pool.jsonl'),('fixed_test','fixed_test.jsonl')]:
        split.write_lines(data/name,[split.as_case(r) for r in selected[part]])
    split.write_lines(data/'fewshot_pool.jsonl',parts['train'])
    _write(data/'temporal_boundaries.json',cuts)
    _write(data/'manifest.json',{'source_files':{},'source_messages':len(source),
        'boundaries_sha256':sha256_file(data/'temporal_boundaries.json')})
    exports=tmp_path/'exports';exports.mkdir()
    spec={'data_ref':'d-0001','source_exports':str(exports),'source_export_files':[],
        'inputs':{},'separation_audit':{'target':'familiar_contacts_later_conversations'}}
    monkeypatch.setattr(audit,'load_weflow_dir',lambda _:source)
    assert audit.audit_sources(spec)['counts']=={'train':1200,'development':1000,'fixed_test':1000}
    rows=split.lines(data/'dev_pool.jsonl');rows[0]['source_span']['start']-=1
    split.write_lines(data/'dev_pool.jsonl',rows)
    with pytest.raises(ValueError,match='context differs'):
        audit.audit_sources(spec)


def test_fresh_paired_runner_draws_each_independent_round_once(priv,tmp_path,case,bundle,monkeypatch):
    cfg,original=bundle
    source=versions.create_judge_version(cfg,{},source_dir=original)
    directory=priv/'judge_training/temporal';directory.mkdir(parents=True)
    artifact=directory/'correction.json'
    correction=json.loads((original/'correction.json').read_text());correction['final_model']['coefficients'][0]=1.
    _write(artifact,correction)
    gen_trace=tracing.CaseTrace(directory,case,{'dataset':'development'});gen_trace.finish('ok')
    generated={'entries':{case['case_id']:{'replies':['九点'],'trace_ref':gen_trace.ref}}}
    spec={'data_ref':'d-0001','generator_ref':'g-0001','created':'now','evaluation_experiment':'fresh-paired',
        'source_judge':source,'training_sha256':'frozen','feature_config':cfg['llm'],'separation_audit':{},
        'protocol':experiment._protocol_snapshot(runner.load_settings())}
    target=job.make_evaluation(directory,spec,artifact,[case],generated)
    FeatureClient.calls=[]
    monkeypatch.setattr(corrected,'CodexJudgeClient',FeatureClient)
    monkeypatch.setattr(rt,'score_formal_judge_pair',lambda model,*a:.8 if model.coefficients[0] else .2)
    pointers=versions.load_pointers()
    runner.run_judge_experiment(target,workers=1)
    assert FeatureClient.calls==[0,1,2]
    record=json.loads((target/'cases.jsonl').read_text())
    assert record['flip_verified'] and len(record['baseline_votes'])==3
    trace=tracing.read(target,record['trace_ref'])
    for i in range(3):
        ops=[op for op in trace['operations'] if op['round']==i]
        draws=[[e['data']['parsed'] for e in op['events'] if e['kind']=='features'][-1] for op in ops]
        assert len(draws)==2 and draws[0]==draws[1]
    row=json.loads((priv/'judge_eval'/('pack-'+directory.name)/'pack.json').read_text())['rows'][0]
    replay=lr.FeatureReplay(None,None,{'llm':cfg['llm']},[row])
    resume=tracing.CaseTrace(target,row,experiment.spec_of(target))
    for i in range(3):
        with resume.operation('judge','baseline',i,{}):replay.get(row,i,FeatureClient(cfg['llm']))
    assert FeatureClient.calls==[0,1,2]
    assert versions.load_pointers()==pointers


def test_finished_with_failures_is_reopened_and_only_failed_cases_retried(tmp_path,monkeypatch):
    target=tmp_path/'evaluation';target.mkdir()
    rows={str(i):{'status':'ok','case_id':str(i)} for i in range(1000)};rows['7']['status']='failed'
    _write(target/'state.json',{'status':'finished'})
    monkeypatch.setattr(job.protocol,'summarize_final_records',lambda _: (rows,0))
    calls=[]
    def run(path,workers):
        assert experiment.state_of(path)['status']=='stopped'
        calls.extend(k for k,r in rows.items() if r['status']=='failed')
        rows['7']['status']='ok'
    monkeypatch.setattr(runner,'run_judge_experiment',run)
    job.evaluate_complete(target,4)
    assert calls==['7']
    assert len(list((target/'snapshots').glob('*.json')))==1


def test_generation_resume_rejects_temporal_metadata_changes():
    cases=[{'case_id':'one','source_span':{'start_timestamp':100},'human_reply':['human']}]
    value={'identity':'frozen','entries':{'one':{'status':'ok','input_sha256':cache.digest(cases[0]),
        'replies':['AI'],'output_sha256':cache.digest(['AI'])}}}
    job.common.check_generations(value,cases,'frozen')
    changed=copy.deepcopy(cases);changed[0]['source_span']['start_timestamp']=101
    with pytest.raises(ConfigError,match='内容或来源'):
        job.common.check_generations(value,changed,'frozen')
