"""Calibrate only existing linear output parameters on eight local NMC cases.

Solve in the SVD coordinates of frozen nonlinear features. Density labels are
the only targets; oracle coefficients are never read. Sampled convex objectives
are optimized jointly across all eight cases, with complete-grid selection and
an independent production-model/decoder audit of the exported checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from density_eqgnn.compact_basis import (CompactBasisConfig, basis_checkpoint_metadata,
    check_basis_checkpoint, radial_to_degree_major)
from density_eqgnn.fixed_grid import FixedGridDensity
from density_eqgnn.frozen_readout import (frozen_digest, promote_linear_readout,
    frozen_features, linear_parameters, set_linear_parameters, blocks_to_degree_major)
from density_eqgnn.models import build_compact_coefficient_model
from density_eqgnn.oracle_basis import FullGridDesign, OracleBasis
from density_eqgnn.periodic import check_periodic_checkpoint, PERIODIC_GEOMETRY_VERSION
from scripts.train_local_nmc_overfit import load_sample, oracle_record, evaluate, predict


def stratified_probes(nearest,core_radius,outer_count,generator):
    """Horvitz--Thompson weights for a full-grid mean, including every core point."""
    core=torch.nonzero(nearest<core_radius,as_tuple=True)[0]
    outer=torch.nonzero(nearest>=core_radius,as_tuple=True)[0]
    selected=outer[torch.randperm(len(outer),device=nearest.device,generator=generator)[:outer_count]]
    ids=torch.cat((core,selected))
    inverse_probability=len(outer)/len(selected) if len(selected) else 0.
    weights=torch.cat((torch.ones(len(core),device=nearest.device,dtype=torch.float64),
        torch.full((len(selected),),inverse_probability,device=nearest.device,dtype=torch.float64)))/len(nearest)
    return ids,weights,len(core)


class ReducedReadout:
    def __init__(self, features, theta, atom_counts, rtol):
        self.theta = theta
        self.radial = theta[0].shape[1]
        self.atom_counts = atom_counts
        self.atom_offsets = np.cumsum([0]+atom_counts)
        self.q, self.inverse, self.base, self.spectra = [], [], [], []
        for degree,f in enumerate(features):
            started = time.perf_counter()
            # LAPACK CPU SVD avoids the slow double Jacobi GPU implementation.
            u,s,vh = torch.linalg.svd(f.cpu(),full_matrices=False)
            keep = s > s[0]*rtol
            q = u[:,keep].to(f.device)
            inverse = (vh[keep].T/s[keep]).to(f.device)
            self.q.append(q)
            self.inverse.append(inverse)
            self.base.append(f@theta[degree])
            record = {'degree':degree,'rows':len(f),'columns':f.shape[1],
                      'rank':int(keep.sum()),'rtol':rtol,
                      'largest_singular':float(s[0]),'smallest_retained':float(s[keep][-1]),
                      'ranks_at_rtol':{str(t):int((s>s[0]*t).sum()) for t in (1e-4,1e-6,1e-8,1e-10,1e-12)},
                      'svd_seconds':time.perf_counter()-started}
            self.spectra.append(record)
            print('SVD '+json.dumps(record),flush=True)
        self.offsets = np.cumsum([0]+[q.shape[1]*self.radial for q in self.q])
        self.size = int(self.offsets[-1])

    def unpack(self,x):
        return [x[self.offsets[l]:self.offsets[l+1]].reshape(q.shape[1],self.radial)
                for l,q in enumerate(self.q)]

    def blocks(self,x,baseline=True):
        return [q@z+(b if baseline else 0) for q,z,b in zip(self.q,self.unpack(x),self.base)]

    def coefficients(self,x,baseline=True):
        blocks = self.blocks(x,baseline)
        result=[]
        for begin,end in zip(self.atom_offsets[:-1],self.atom_offsets[1:]):
            result.append(blocks_to_degree_major([b[begin*(2*l+1):end*(2*l+1)]
                          for l,b in enumerate(blocks)],self.radial).flatten())
        return result

    def adjoint(self,coefficients):
        all_blocks=[[] for _ in self.q]
        for count,c in zip(self.atom_counts,coefficients):
            shaped=c.reshape(count,-1)
            offset=0
            for degree in range(len(self.q)):
                width=2*degree+1
                all_blocks[degree].append(shaped[:,offset:offset+self.radial*width].reshape(count,self.radial,width).transpose(1,2).reshape(-1,self.radial))
                offset+=self.radial*width
        return torch.cat([(q.T@torch.cat(b).double()).flatten() for q,b in zip(self.q,all_blocks)])

    def parameters(self,x):
        return [t+v@z for t,v,z in zip(self.theta,self.inverse,self.unpack(x))]


class DensityOperator:
    def __init__(self, reduced, designs, normalizers, row_weights=None):
        self.reduced,self.designs,self.normalizers = reduced,designs,normalizers
        self.row_weights=row_weights

    def forward(self,x,baseline=False):
        result=torch.cat([(a@c.float()).double()/n for a,c,n in
                          zip(self.designs,self.reduced.coefficients(x,baseline),self.normalizers)])
        return result if self.row_weights is None else result*self.row_weights

    def adjoint(self,y):
        if self.row_weights is not None:
            y=y*self.row_weights
        start=0
        coefficients=[]
        for a,n in zip(self.designs,self.normalizers):
            stop=start+a.shape[0]
            coefficients.append(a.T@(y[start:stop]/n).float())
            start=stop
        return self.reduced.adjoint(coefficients)


class DensityMatvec(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,operator):
        ctx.operator=operator
        return operator.forward(x,baseline=True)

    @staticmethod
    def backward(ctx,g):
        return ctx.operator.adjoint(g),None


@torch.no_grad()
def cgls(operator,target,x,steps):
    x=x.clone()
    residual=target-operator.forward(x,baseline=True)
    s=operator.adjoint(residual)
    p=s.clone()
    gamma=s.square().sum()
    initial=float(gamma.sqrt())
    history=[]
    for step in range(1,steps+1):
        q=operator.forward(p)
        alpha=gamma/q.square().sum().clamp_min(1e-30)
        x+=alpha*p
        residual-=alpha*q
        if step%50==0:
            residual=target-operator.forward(x,baseline=True)
        s=operator.adjoint(residual)
        next_gamma=s.square().sum()
        if step%50==0 or step==steps:
            record={'step':step,'sampled_nmae':float(residual.abs().mean()),
                    'relative_normal_residual':float(next_gamma.sqrt())/max(initial,1e-30)}
            history.append(record)
            print('CGLS '+json.dumps(record),flush=True)
        if float(next_gamma.sqrt()) < initial*1e-7:
            break
        # Restart after explicit residual replacement in finite precision.
        p=s if step%50==0 else s+(next_gamma/gamma)*p
        gamma=next_gamma
    return x,history


class CachedCoefficientModel:
    def __init__(self,samples,coefficients,basis):
        self.values={s['model_positions'].data_ptr():c.reshape(len(s['numbers']),basis.num_radial,-1)
                     for s,c in zip(samples,coefficients)}
        self.basis=basis

    def eval(self):
        return self

    def __call__(self,numbers,positions,mask,cell):
        # Input is degree-major flattened, not a radial-major reshaping.
        flat=self.values[positions.data_ptr()].reshape(len(numbers[0]),-1)
        blocks=[]
        for degree in range(self.basis.lmax+1):
            begin=self.basis.num_radial*degree**2
            end=self.basis.num_radial*(degree+1)**2
            blocks.append(flat[:,begin:end].reshape(-1,self.basis.num_radial,2*degree+1))
        return {'coefficients':torch.cat(blocks,dim=-1).flatten(1)[None]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--probes',type=int,default=32768)
    parser.add_argument('--core-radius',type=float,default=.35,
                        help='Keep every grid point inside this nearest-atom radius; sample remaining points with importance weights')
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--iterations',type=int,default=300)
    parser.add_argument('--cg-steps',type=int,default=400)
    parser.add_argument('--svd-rtol',type=float,default=1e-8)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--chunk',type=int,default=16384)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.rounds<1 or args.probes<1 or args.iterations<1 or args.core_radius<0 or not 0<args.svd_rtol<1:
        parser.error('Invalid solver settings')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda')
    start=time.perf_counter()
    state=torch.load(args.parent/'best.pt',map_location=device,weights_only=False)
    parent_result=json.loads((args.parent/'result.json').read_text())
    check_periodic_checkpoint(state,pbc=True,resume=True)
    basis=CompactBasisConfig(**state['basis_config'])
    check_basis_checkpoint(state,basis)
    model=build_compact_coefficient_model(state['model_config']).to(device)
    model.load_state_dict(state['model'])
    digest=frozen_digest(model)
    promote_linear_readout(model)
    model.eval()
    variant=parent_result['variant']
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
              if k not in ('prepare_only','resume')}
    settings.update(parent_sha256=hashlib.sha256((args.parent/'best.pt').read_bytes()).hexdigest(),
        frozen_sha256=digest,basis_config=state['basis_config'],model_config=state['model_config'],
        interpretation='Existing linear readout and element baseline only; frozen nonlinear features; local training-set diagnostic',
        objective='Importance-weighted macro density L1; all core points plus uniform outer subset; no coefficient targets; full-grid selection')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if (args.output_dir/'settings.json').exists():
        if not args.resume or json.loads((args.output_dir/'settings.json').read_text())!=settings:
            raise ValueError('Existing output requires --resume with identical settings')
    (args.output_dir/'settings.json').write_text(json.dumps(settings,indent=2))
    samples=[]
    for i in range(8):
        s=load_sample(Path('data/nmc_cache')/f'{i}.CHGCAR.lz4',device)
        s.update(id=i,model_positions=s['positions'].float()[None],model_cell=s['cell'].float()[None],
                 atom_mask=torch.ones(1,len(s['positions']),dtype=torch.bool,device=device))
        path,s['oracle']=oracle_record(i,variant)
        if s['oracle']['sha256']!=s['metadata']['sha256']:
            raise ValueError('DFT source differs from oracle')
        s['oracle_path']=str(path)
        s['normalizer']=s['target'].abs().mean()
        samples.append(s)
    species=torch.unique(torch.cat([s['numbers'] for s in samples]))
    cache=args.output_dir/'feature_cache.pt'
    if cache.exists():
        saved=torch.load(cache,map_location=device,weights_only=False)
        if saved['frozen_sha256']!=digest:
            raise ValueError('Frozen features differ')
        reduced=saved['reduced']
    else:
        fs=[frozen_features(model,s['numbers'][None],s['model_positions'],s['atom_mask'],s['model_cell'],species) for s in samples]
        features=[torch.cat([f[l] for f in fs]) for l in range(basis.lmax+1)]
        theta=linear_parameters(model,species)
        reduced=ReducedReadout(features,theta,[len(s['numbers']) for s in samples],args.svd_rtol)
        x=torch.zeros(reduced.size,device=device,dtype=torch.float64)
        errors=[]
        for s,c in zip(samples,reduced.coefficients(x)):
            direct=radial_to_degree_major(predict(model,s),basis).flatten()
            errors.append(float((direct-c).abs().max()))
        if max(errors)>1e-9:
            raise ValueError(f'Frozen feature map differs from model: {errors}')
        torch.save({'reduced':reduced,'frozen_sha256':digest,'species':species,
                    'initial_coefficient_errors':errors},cache)
        del features,fs
    (args.output_dir/'feature_spectrum.json').write_text(json.dumps(reduced.spectra,indent=2))
    print(f'PREPARED parameters={reduced.size} elapsed={time.perf_counter()-start:.1f}',flush=True)
    if args.prepare_only:
        return
    for s in samples:
        s['decoder']=FixedGridDensity(s['positions'],s['probes'],s['cell'],basis)
    (args.output_dir/'sources.json').write_text(json.dumps([
        {'sample_id':s['id'],**s['metadata'],'oracle_file':s['oracle_path']} for s in samples],indent=2))
    @torch.no_grad()
    def full_grid(x,independent=False):
        return evaluate(CachedCoefficientModel(samples,reduced.coefficients(x),basis),samples,basis,args.chunk,independent=independent)
    history=[]
    x=torch.zeros(reduced.size,device=device,dtype=torch.float64)
    best_x=x.clone()
    best=parent_result['macro_nmae']
    completed_round=0
    if (args.output_dir/'solver.pt').exists():
        previous=torch.load(args.output_dir/'solver.pt',map_location=device,weights_only=False)
        x,best_x,best,history,completed_round=(previous[k] for k in ('x','best_x','best','history','completed_round'))
    else:
        initial=full_grid(x)
        best=initial['macro_nmae']
        history.append({'stage':'initial_promoted_readout',**initial})
        print('EVAL '+json.dumps({'stage':'initial','macro_nmae':best}),flush=True)
    def record(x,stage,**extra):
        nonlocal best,best_x
        metrics=full_grid(x)
        history.append({'stage':stage,**extra,**metrics})
        if metrics['macro_nmae']<best:
            best=metrics['macro_nmae']; best_x=x.detach().clone()
        (args.output_dir/'history.json').write_text(json.dumps(history,indent=2))
        print('EVAL '+json.dumps({'stage':stage,'macro_nmae':metrics['macro_nmae'],'best':best,**extra}),flush=True)
    for round_index in range(completed_round+1,args.rounds+1):
        generator=torch.Generator(device=device).manual_seed(args.seed+40000+round_index)
        designs=[]; targets=[]; probabilities=[]; sampling=[]
        for s in samples:
            ids,probability,core_count=stratified_probes(s['decoder'].nearest_atom,args.core_radius,args.probes,generator)
            # Each structure contributes 1/8 of the complete-grid objective.
            probabilities.append(probability/len(samples))
            sampling.append({'sample_id':s['id'],'core_points':core_count,'outer_points':len(ids)-core_count,
                'indices_sha256':hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()})
            designs.append(FullGridDesign(OracleBasis([(s['positions'],basis.radial_spec)],s['cell']),s['probes'][ids]))
            targets.append(s['target'][ids]/s['normalizer'])
        target=torch.cat(targets)
        weights=torch.cat(probabilities)*len(target)
        raw_target=target
        operator=DensityOperator(reduced,designs,[s['normalizer'] for s in samples],weights)
        target=raw_target*weights
        print(f'ROUND {round_index} memory_GB={torch.cuda.memory_allocated()/1e9:.2f}',flush=True)
        if round_index==1 and args.cg_steps:
            ls_operator=DensityOperator(reduced,designs,[s['normalizer'] for s in samples],weights.sqrt())
            x,cg_history=cgls(ls_operator,raw_target*weights.sqrt(),x,args.cg_steps)
            record(x,'least_squares',solver_history=cg_history,sampling=sampling)
            del ls_operator
        x=torch.nn.Parameter(x)
        epsilon=1e-4 if round_index==1 else 1e-5
        optimizer=torch.optim.LBFGS([x],lr=1.,max_iter=100,max_eval=200,history_size=40,
            line_search_fn='strong_wolfe',tolerance_grad=1e-11,tolerance_change=1e-13)
        calls=0
        def closure():
            nonlocal calls
            optimizer.zero_grad(set_to_none=True)
            residual=DensityMatvec.apply(x,operator)-target
            loss=torch.sqrt(residual.square()+epsilon**2).mean()
            loss.backward()
            if not torch.isfinite(loss) or not torch.isfinite(x.grad).all():
                raise RuntimeError('Nonfinite linear solve')
            calls+=1
            if calls%50==0:
                print(f'FIT round={round_index} closure={calls} loss={float(loss.detach()):.8f} grad={float(x.grad.abs().max()):.3g}',flush=True)
            return loss
        for iteration in range(0,args.iterations,100):
            optimizer.param_groups[0]['max_iter']=min(100,args.iterations-iteration)
            optimizer.step(closure)
            with torch.no_grad():
                residual=operator.forward(x,baseline=True)-target
                gradient=operator.adjoint(residual/torch.sqrt(residual.square()+epsilon**2))/len(target)
            record(x,f'round_{round_index}_iteration_{min(iteration+100,args.iterations)}',
                sampled_nmae=float(residual.abs().mean()),gradient_inf=float(gradient.abs().max()),
                closures=calls,epsilon=epsilon,sampling=sampling)
        x=x.detach()
        torch.save({'x':x,'best_x':best_x,'best':best,'history':history,'completed_round':round_index},args.output_dir/'solver.pt')
        del operator,designs,targets,optimizer,residual,gradient,weights,raw_target,probabilities
        torch.cuda.empty_cache()
    parameters=reduced.parameters(best_x)
    set_linear_parameters(model,species,parameters)
    if frozen_digest(model)!=digest:
        raise RuntimeError('Frozen weights changed')
    coefficient_errors=[]
    for s,c in zip(samples,reduced.coefficients(best_x)):
        direct=radial_to_degree_major(predict(model,s),basis).flatten()
        coefficient_errors.append(float((direct-c).abs().max()))
    final=evaluate(model,samples,basis,args.chunk,independent=True)
    final.update(variant=variant,parent_macro_nmae=parent_result['macro_nmae'],
        reduced_map_macro_nmae=best,export_max_coefficient_difference=max(coefficient_errors),
        frozen_parameters_unchanged=True,frozen_sha256=digest,feature_spectrum=reduced.spectra,
        reduced_parameter_count=reduced.size,linear_parameter_rms=[float(p.square().mean().sqrt()) for p in parameters],
        basis_config=state['basis_config'],model_config=state['model_config'],seed=args.seed,
        interpretation='Finite-budget linear calibration on eight training structures, not a certified capacity floor',
        elapsed_seconds=time.perf_counter()-start)
    torch.save({'model':model.state_dict(),'model_config':state['model_config'],
        **basis_checkpoint_metadata(basis),'periodic_geometry_version':PERIODIC_GEOMETRY_VERSION,
        'linear_readout_dtype':'float64','coefficient_scale':torch.ones(basis.num_radial,basis.angular_channels),
        'frozen_sha256':digest,'parent':str(args.parent),'calibration_settings':settings},args.output_dir/'best.pt')
    (args.output_dir/'result.json').write_text(json.dumps(final,indent=2))
    print('RESULT '+json.dumps({k:v for k,v in final.items() if k!='records'}),flush=True)


if __name__=='__main__':
    main()
