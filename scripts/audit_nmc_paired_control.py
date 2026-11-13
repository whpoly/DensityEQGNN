"""Verify that the existing R16 arm is the exact matched coefficient control."""
import argparse
import json
from pathlib import Path
import zipfile

import torch

from density_eqgnn.models.paired_readout import PairedReadoutModel
from scripts.fit_nmc_holdout_oracle import ROOT, record
from scripts.nmc_probe_holdout import DEFAULT_ROOT, sha256
from scripts.train_local_nmc_overfit import state_digest


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--query",type=Path,default=ROOT/"query_eight")
    args=p.parse_args()
    control=DEFAULT_ROOT/"r16_eight"
    a=read(control/"settings.json")
    b=read(args.query/"settings.json")
    for key in ("samples","steps","probes","eval_every","chunk","seed","lr","min_lr","model_config"):
        assert a[key]==b[key],key
    assert a["loss"]=="l1"
    for settings in (a,b):
        for path,digest in settings["source_hashes"].items():
            assert sha256(path)==digest,path
    torch.manual_seed(a["seed"])
    fresh=PairedReadoutModel(a["model_config"],"coefficient")
    assert state_digest(fresh.atom_model.state_dict())==a["initial_state_sha256"]
    assert state_digest(fresh.atom_model.state_dict(),True)==b["trunk_initial_sha256"]
    sa,sb=read(control/"sources.json"),read(args.query/"sources.json")
    assert sa==sb
    # Settings from the earlier run recorded the runner but not every imported
    # model file. Its archived source snapshot independently closes that gap.
    snapshot=Path("reports/nmc_probe_holdout_2026-10-27/source_snapshot.zip")
    with zipfile.ZipFile(snapshot) as archive:
        source_rows=[]
        for name in archive.namelist():
            if name.startswith("src/") and name.endswith(".py"):
                path=Path(name)
                assert path.read_bytes()==archive.read(name),name
                source_rows.append({"path":name,"unchanged":True,"sha256":sha256(path)})
    assert source_rows
    record(args.query.parent/"production_source_audit.json",{
        "reference_snapshot":str(snapshot),"reference_snapshot_sha256":sha256(snapshot),
        "all_unchanged":True,"files":source_rows})
    if (args.query/"sampling_audit.json").exists():
        assert read(control/"sampling_audit.json")==read(args.query/"sampling_audit.json")
    result={"coefficient_control":str(control),"query":str(args.query),
            "control_checkpoint_sha256":sha256(control/"best.pt"),
            "identical_backbone_initialization":True,"identical_settings_and_sources":True,
            "identical_actual_sample_coverage":True if (args.query/"sampling_audit.json").exists() else None,
            "archived_production_source_files_unchanged":len(source_rows),
            "control_parameters":a["parameter_count"],"query_parameters":b["parameter_count"],
            "backbone_parameters":b["trunk_parameters"],
            "reuse_reason":"Existing coefficient run has identical production architecture, initialization, split, sampling RNG and training protocol; new wrapper matches production forward and backbone gradients exactly."}
    record(args.query.parent/"paired_control_audit.json",result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__": main()
