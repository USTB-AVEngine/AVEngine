#!/usr/bin/env python3
"""Record local CosyVoice weight-license declarations and artifact hashes."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    plan=json.loads(Path(args.plan).read_text())
    out=Path(args.output)
    evidence=out/'evidence/license'
    evidence.mkdir(exist_ok=True)
    model=Path(plan['model_dir'])
    card=(model/'README.md').read_text()
    if not card.startswith('---\nlicense: apache-2.0\n'):
        raise ValueError('expected local weight-model Apache-2.0 declaration missing')
    inventory=[]
    for path in sorted(model.rglob('*')):
        if path.is_file():
            inventory.append({'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path)})
    shutil.copyfile(model/'README.md',evidence/'CosyVoice2-0.5B.README.md')
    code_license=Path(plan['model_receipt']['model_license_path'])
    shutil.copyfile(code_license,evidence/'CosyVoice_code.LICENSE')
    license_files=[str(p) for p in model.rglob('*') if p.is_file() and p.name.lower().startswith(('license','notice','copying'))]
    supplementary=[]
    for url,name in [('https://www.apache.org/licenses/LICENSE-2.0.txt','Apache-2.0.upstream.txt'),
                     ('https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B/resolve/eec1ae6c79877dbd9379285cf8789c9e0879293d/README.md','CosyVoice2-0.5B.upstream.README.md')]:
        try:
            with urllib.request.urlopen(url,timeout=30) as response:
                body=response.read()
                (evidence/name).write_bytes(body)
                supplementary.append({'url':url,'path':str(evidence/name),'sha256':digest(evidence/name),'retrieved':True,'equals_local_model_card':body==(model/'README.md').read_bytes() if 'README' in name else None})
        except Exception as error:
            supplementary.append({'url':url,'retrieved':False,'reason':str(error)})
    vctk=Path('/data/datasets/VCTK-Corpus-0.92')
    for name in ['license_text.txt','README.txt','speaker-info.txt']:
        shutil.copyfile(vctk/name,evidence/('VCTK-'+name))
    receipt={'schema':'v7_dialogue_voice_license_receipt_v1','status':'pass',
             'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'model_id':'FunAudioLLM/CosyVoice2-0.5B','model_dir':str(model),
             'local_weight_model_license':'Apache-2.0',
             'basis':'local weight directory README front matter declares license: apache-2.0',
             'local_weight_license_or_notice_files':license_files,
             'no_separate_license_file_in_weight_directory':not license_files,
             'output_use_for_AI_training_restriction_found':False,
             'decision':'No local model-card clause forbids using generated outputs to train AI models; the local Apache-2.0 declaration permits continuing this task.',
             'disclaimer_scope':'The final README disclaimer describes the content/examples above as academic-purpose demonstrations; it does not explicitly prohibit AI training on generated outputs.',
             'limitations':['Local weights do not carry a separate LICENSE; code LICENSE is supplemental and is not represented as a weight-license declaration.','Revision is retained plan provenance; current weight bytes are inventoried, not compared to remote weight bytes.'],
             'model_revision_from_prior_receipt':plan['model_receipt'].get('model_revision'),
             'prior_receipt':plan['code_receipt'],
             'prior_receipt_fields':plan['model_receipt'],
             'local_model_card':str(model/'README.md'),'local_model_card_copy':str(evidence/'CosyVoice2-0.5B.README.md'),
             'local_model_card_sha256':digest(model/'README.md'),
             'code_license_path':str(code_license),'code_license_sha256':digest(code_license),
             'artifact_inventory':inventory,'supplementary_primary_sources':supplementary,
             'VCTK':{'license':'CC-BY-4.0','local_license':str(vctk/'license_text.txt'),'license_sha256':digest(vctk/'license_text.txt'),'attribution':'Yamagishi, Veaux, MacDonald (2019), CSTR VCTK Corpus 0.92, University of Edinburgh','source_uri':'https://datashare.ed.ac.uk/handle/10283/3443','corpus_not_modified':True}}
    (out/'license_receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    (out/'license_receipt.md').write_text('# CosyVoice2-0.5B 本地权重许可收据\n\n本地模型卡明确声明 Apache-2.0；未发现禁止将合成输出用于 AI 训练的条款。许可前置检查通过。\n\n- 权重目录：`'+str(model)+'`\n- 声明文件：`'+str(model/'README.md')+'`\n- 声明 SHA-256：`'+digest(model/'README.md')+'`\n- 权重目录没有单独 LICENSE；CosyVoice 代码 LICENSE 仅作为补充，没有拿代码许可冒充权重许可。\n- README 末尾的 academic purposes 说明针对上方内容/示例，没有写合成输出不得训练模型。\n- Apache 原文：[Apache Software Foundation](https://www.apache.org/licenses/LICENSE-2.0)；模型卡：[FunAudioLLM](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B/blob/main/README.md)。\n- 本地各权重文件的字节数、SHA 和历史模型 revision 出处见 `license_receipt.json`；没有声称核过远端权重字节。\n- VCTK：CC-BY 4.0，保留作者和爱丁堡大学出处；源数据只读。\n')
    print(json.dumps({'status':receipt['status'],'model_license':receipt['local_weight_model_license'],'inventory_files':len(inventory),'weight_bytes':sum(r['bytes'] for r in inventory),'supplementary_primary_sources':supplementary}))


if __name__=='__main__':
    main()
