"""Device-side ON/OFF candidates, native support and periodic templates.

Same event-time/discovery/validation rules as the existing CUDA reference.
No NN or grey reconstruction. Output is sparse; diagnostic maps stay off IPC.
"""
import numpy as np
from predictive_harmonics import BASE_FREQUENCIES, PARAMS


class DeviceRotorFront:
    def __init__(self, width=1280, height=720, fused_sparse_copy=True, unique_harmonics=False, fused_harmonics=False, native_preprocessing=False,warp_reduction=True,native_spatial=False,regional_seeds=False,hybrid_histogram=False,weak_regional=False,phase_statistics=False):
        import torch
        self.torch = torch; self.width = width; self.height = height
        self.size = width*height
        self.fused_sparse_copy = bool(fused_sparse_copy)
        self.unique_harmonics = bool(unique_harmonics)
        self.native_preprocessing=bool(native_preprocessing)
        self.warp_reduction=bool(warp_reduction)
        self.native_spatial=bool(native_spatial)
        self.weak_regional=bool(weak_regional)
        self.regional_seeds=bool(regional_seeds or weak_regional)
        self.hybrid_histogram=bool(hybrid_histogram)
        self.phase_statistics=bool(phase_statistics)
        if self.phase_statistics and not self.native_preprocessing:raise ValueError('Spatial Fourier statistics require native timestamps')
        if self.weak_regional and self.hybrid_histogram:raise ValueError('Weak regional candidates use the native GPU histogram')
        if self.regional_seeds and not self.native_preprocessing:raise ValueError('Regional evidence requires native per-pixel held-out statistics')
        self.fused_harmonics=bool(fused_harmonics or native_preprocessing);self.fused_kernel=None
        self.f = torch.as_tensor(BASE_FREQUENCIES, dtype=torch.float32, device='cuda')
        self.h = torch.arange(1., 5., device='cuda')
        products=BASE_FREQUENCIES[:,None]*np.arange(1.,5.)[None]
        unique,inverse=np.unique(products,return_inverse=True)
        self.spectral_f=torch.as_tensor(unique,dtype=torch.float32,device='cuda')
        self.harmonic_index=torch.as_tensor(inverse.reshape(products.shape),dtype=torch.long,device='cuda')
        x = torch.arange(-1., 2., device='cuda')
        g = torch.exp(-.5*(x/.8)**2); self.g = g/g.sum()

    def gaussian(self, x):
        import torch.nn.functional as F
        x = F.conv2d(F.pad(x, (1,1,0,0), mode='reflect'), self.g.reshape(1,1,1,3))
        return F.conv2d(F.pad(x, (0,0,1,1), mode='reflect'), self.g.reshape(1,1,3,1))

    def morphology(self, x, size, erode=False):
        import torch.nn.functional as F
        if erode: x = -x
        left = size//2; right = size-1-left
        x = F.max_pool2d(F.pad(x, (left,right,left,right), value=-float('inf')), size, stride=1)
        return -x if erode else x

    def process(self, events, cutoff_s, window_us=33333):
        import torch.nn.functional as F
        T = self.torch; end = round(cutoff_s*1e6); begin = end-window_us
        cpu_activity=None
        if self.native_preprocessing:
            if self.fused_kernel is None:
                from rotor_nvrtc import FusedHarmonics
                self.fused_kernel=FusedHarmonics(warp_reduction=self.warp_reduction)
            if self.hybrid_histogram and len(events)>=500000 and window_us==33333:
                on,off,pixels,lengths,nt,et,polarity,cpu_activity,selected_events=self.fused_kernel.prepare_cpu_histogram(events,begin,end,self.width,self.height)
                total=on+off;n=len(pixels)
            else:
                on,off,total,local_sum,dense,pixels,lengths,nt,et,polarity,counts20=self.fused_kernel.prepare(events,begin,end,self.width,self.height,minimum_half=3 if self.weak_regional else 6)
                local=local_sum.float()/961.;n=len(pixels)
                selected_events=None
        else:
            # One packed transfer; preserve every timestamp at every selected pixel.
            packed = np.column_stack([events[k] for k in ('x','y','p','t')]).astype(np.int64)
            e = T.as_tensor(packed, device='cuda')
            e = e[(e[:,3] >= begin)&(e[:,3] < end)]
            pix = e[:,1]*self.width+e[:,0]; times = e[:,3]-begin
            half = times*2 >= window_us
            counts = T.bincount(pix+(e[:,2]>0).long()*self.size, minlength=2*self.size).reshape(2,-1)
            off, on = counts[0], counts[1]
            total = on+off
            image = total.reshape(self.height,self.width)
            # Exact integer sums, including replicated borders. Average-pool then
            # multiply had a rounding difference at one real threshold equality.
            ix=T.arange(-15,self.width+15,device='cuda').clamp(0,self.width-1)
            iy=T.arange(-15,self.height+15,device='cuda').clamp(0,self.height-1)
            cs=T.cat([T.zeros((self.height,1),dtype=T.int64,device='cuda'),image[:,ix].cumsum(1)],1)
            sx=cs[:,31:]-cs[:,:-31]
            cs=T.cat([T.zeros((1,self.width),dtype=T.int64,device='cuda'),sx[iy].cumsum(0)],0)
            local_sum=(cs[31:]-cs[:-31]).reshape(-1)
            local=local_sum.float()/961.
            dense=(total>=6)&(total*961>=2*local_sum)
            hc = T.bincount(pix+half.long()*self.size,minlength=2*self.size).reshape(2,-1)
            eligible = dense&(hc >= PARAMS['min_events_per_half']).all(0)
            keep = eligible[pix]
            order = T.argsort(pix[keep], stable=True)
            ep = pix[keep][order]; et = times[keep][order]; polarity = e[:,2][keep][order]
            pixels, lengths = T.unique_consecutive(ep, return_counts=True)
            n = len(pixels)
            selected_events=len(e)
        mask = T.zeros(self.size, dtype=T.bool, device='cuda')
        if n and self.fused_harmonics:
            if self.fused_kernel is None:
                from rotor_nvrtc import FusedHarmonics
                self.fused_kernel=FusedHarmonics(warp_reduction=self.warp_reduction)
            periodic=self.fused_kernel(et,polarity.to(T.int16).contiguous(),pixels,lengths,nt if self.native_preprocessing else hc[0,pixels],self.f,window_us)
            if self.weak_regional:
                # Weak pixels may jointly seed a region, but must never become
                # individually confirmed pixels or fabricated per-motor RPM.
                strict_eligible=(nt>=6)&(lengths-nt>=6)
                periodic[:,5]*=strict_eligible;periodic[:,1]*=strict_eligible
            mask[pixels]=periodic[:,5]>0
        elif n:
            ids = T.repeat_interleave(T.arange(n, device='cuda'), lengths)
            train = et*2 < window_us
            it, iv = ids[train], ids[~train]
            nt, nv = hc[0,pixels], hc[1,pixels]
            ntf, nvf = nt.float(), nv.float()
            first_t = T.cumsum(nt,0)-nt; first_v = T.cumsum(nv,0)-nv
            st, sv = et[train], et[~train]
            coverage = (st[first_t+nt-1]-st[first_t] >= window_us/4)&(sv[first_v+nv-1]-sv[first_v] >= window_us/4)
            occupancy = T.bincount(ids*8+(et*8//window_us).clamp_max(7),minlength=n*8).reshape(n,2,4)>0
            coverage &= (occupancy.sum(2) >= 2).all(1)
            tp = 2*polarity[train].float()-1; vp = 2*polarity[~train].float()-1
            mean = T.segment_reduce(tp,'sum',lengths=nt,unsafe=True)/ntf
            weights = tp-mean[it]
            if self.unique_harmonics:
                # Neighboring combs share many Fourier coefficients. Evaluate
                # each physical frequency once, then gather the same 57x4 bank.
                phase=st.float()[:,None]/1e6*(2*T.pi*self.spectral_f[None])
                real=T.segment_reduce(weights[:,None]*T.cos(phase),'sum',lengths=nt,unsafe=True)/ntf[:,None]
                imag=T.segment_reduce(-weights[:,None]*T.sin(phase),'sum',lengths=nt,unsafe=True)/ntf[:,None]
                zr=real[:,self.harmonic_index];zi=imag[:,self.harmonic_index]
            else:
                phase = st.float()[:,None,None]/1e6*(2*T.pi*self.f[None,:,None]*self.h[None,None,:])
                zr = T.segment_reduce(weights[:,None,None]*T.cos(phase),'sum',lengths=nt,unsafe=True)/ntf[:,None,None]
                zi = T.segment_reduce(-weights[:,None,None]*T.sin(phase),'sum',lengths=nt,unsafe=True)/ntf[:,None,None]
            power = (zr*zr+zi*zi).sum(2); c0 = power.argmax(1)
            separated = T.abs(T.log(self.f[None]/self.f[c0,None]))>.2
            c1 = T.where(separated,power,-T.ones_like(power)).argmax(1)
            choices = T.stack([c0,c1],1); ar = T.arange(n,device='cuda')[:,None]
            cf = self.f[choices]; real, imag = zr[ar,choices], zi[ar,choices]
            phase = 2*T.pi*sv.float()[:,None,None]/1e6*cf[iv,:,None]*self.h[None,None]
            predicted = (real[iv]*T.cos(phase)-imag[iv]*T.sin(phase)).sum(2)
            mean = T.segment_reduce(predicted,'sum',lengths=nv,unsafe=True)/nvf[:,None]
            centered = predicted-mean[iv]
            projection = T.segment_reduce(vp[:,None]*centered,'sum',lengths=nv,unsafe=True)
            energy = T.segment_reduce(centered*centered,'sum',lengths=nv,unsafe=True)
            score = T.where(projection>0,projection*projection/(2*energy.clamp_min(1e-12)),0.)
            bound = (2*T.exp(-score)).clamp_max(1.)
            choice = score.argmax(1); ar1 = T.arange(n,device='cuda')
            accepted = coverage&(bound.min(1).values <= PARAMS['conditional_random_sign_bound'])
            frequency = T.where(accepted,cf[ar1,choice],0.)
            mask[pixels] = accepted
            periodic = T.stack([pixels.float(),frequency,score[ar1,choice],cf[:,0],cf[:,1],accepted.float()],1)
        else:
            periodic = T.empty((0,12 if self.fused_harmonics else 6),device='cuda')
        accepted_count=int(mask.sum())
        regional_groups=regional_pixels=0;discovery_mask=mask
        if self.regional_seeds and n and accepted_count<12:
            # Fixed IMAGE cells: membership and the frequency hypotheses come
            # from timestamps/discovery, without a body box chosen from the
            # validation signs. Sum held-out evidence, not raw pixel phases.
            cell_px=64;columns=(self.width+cell_px-1)//cell_px;rows=(self.height+cell_px-1)//cell_px
            nf=len(self.f);ng=columns*rows*nf
            cells=(pixels//self.width//cell_px)*columns+(pixels%self.width//cell_px)
            bins=T.round((periodic[:,3:5]-self.f[0])/15.).long().clamp(0,nf-1)
            ids=(cells[:,None]*nf+bins).reshape(-1)
            projection=periodic[:,[6,8]].reshape(-1);energy=periodic[:,[7,9]].reshape(-1)
            sums=T.zeros((3,ng),dtype=T.float32,device='cuda')
            sums[0].scatter_add_(0,ids,projection);sums[1].scatter_add_(0,ids,energy)
            sums[2].scatter_add_(0,ids,(energy>0).float())
            score=T.where(sums[0]>0,sums[0]**2/(2*sums[1].clamp_min(1e-12)),0.)
            good=(sums[2]>=3)&(score>=float(np.log(2*ng/.01)))
            regional_groups=int(good.sum())
            if regional_groups:
                selected=(good[ids]&(energy>0)).reshape(-1,2).any(1)
                regional_pixels=int(selected.sum())
                extra=T.zeros_like(mask);extra[pixels]=selected;discovery_mask=mask|extra
        native=T.empty((0,2),device="cuda")
        if (accepted_count or regional_pixels) and self.native_preprocessing and self.native_spatial:
            support,seed=self.fused_kernel.spatial_support(off,discovery_mask,self.g,self.width,self.height)
            mp=T.nonzero(support).flatten();native=T.stack([mp.float(),seed[mp].float()],1)
        elif accepted_count or regional_pixels:
            shape = (1,1,self.height,self.width)
            feature = T.minimum(self.gaussian(on.float().reshape(shape)),self.gaussian(off.float().reshape(shape)))
            feature = T.where(total.reshape(shape) >= 2, feature, 0.)
            support = (self.gaussian(feature) >= 4.).float()
            support = self.morphology(self.morphology(support,2,True),2)
            support = self.morphology(self.morphology(support,3),3,True).reshape(-1)>0
            seed = discovery_mask.reshape(shape)
            for _ in range(2):
                pad = F.pad(seed.float(),(1,1,1,1))
                seed = (pad[:,:,1:-1,1:-1]+pad[:,:,:-2,1:-1]+pad[:,:,2:,1:-1]+pad[:,:,1:-1,:-2]+pad[:,:,1:-1,2:])>0
            seed = seed.reshape(-1)&support
            # Return the sparse propagation mask even at non-event bridging pixels.
            mp = T.nonzero(support).flatten()
            native = T.stack([mp.float(),seed[mp].float()],1)
        if self.phase_statistics:
            phase=self.fused_kernel.phase_statistics(et,polarity,lengths,nt,periodic) if n else T.empty((0,4),dtype=T.float32,device='cuda')
            periodic=T.cat([periodic,phase],1)
        if cpu_activity is None:
            ap = T.nonzero(total).flatten()
            activity = T.stack([ap.float(),on[ap].float(),off[ap].float(),local[ap],dense[ap].float()],1)
            if self.native_preprocessing:
                activity=T.cat([activity,counts20[1,ap,None].float(),counts20[0,ap,None].float()],1)
        else:activity=T.empty((0,7),dtype=T.float32,device='cuda')
        if self.fused_sparse_copy:
            # One device-to-host transfer rather than three blocking copies.
            # Returned views share the host buffer and retain their exact shape.
            sizes = [a.numel() for a in (activity, periodic, native)]
            host = T.cat([a.reshape(-1) for a in (activity, periodic, native)]).cpu().numpy()
            first, second = sizes[0], sizes[0]+sizes[1]
            outputs = (host[:first].reshape(-1,activity.shape[1]), host[first:second].reshape(-1,periodic.shape[1]), host[second:].reshape(-1,2))
        else:
            outputs = (activity.cpu().numpy(), periodic.cpu().numpy(), native.cpu().numpy())
        if cpu_activity is not None:outputs=(cpu_activity,outputs[1],outputs[2])
        if self.native_preprocessing:
            # Native counts already contain the exact accepted timestamp/XY
            # range. Reuse their small HOST sparse sum instead of rescanning a
            # million strided CPU timestamps while CUDA could be progressing.
            selected_events=int(outputs[0][:,1:3].sum(dtype=np.float64))
        return *outputs, dict(
            events=selected_events,begin_us=begin,end_us=end,window_us=window_us,
            accepted_pixels=accepted_count,eligible_pixels=n,
            regional_seed_groups=regional_groups,regional_seed_pixels=regional_pixels,
            regional_evidence_is_not_individual_pixel_confirmation=True,
            parameters=dict(PARAMS,window_us=window_us,device_preprocessing=True,
                unique_harmonic_coefficients=self.unique_harmonics,
                fused_native_spectrum=self.fused_harmonics,native_preprocessing=self.native_preprocessing,
                warp_reduction=self.warp_reduction,
                native_spatial=self.native_spatial,
                regional_seeds=self.regional_seeds,
                weak_regional_templates=self.weak_regional,
                native_train_validation_first_complex_coefficients=self.phase_statistics,
                histogram_backend='CPU_OpenCV_joint_sufficient_statistics' if cpu_activity is not None else 'CUDA_native' if self.native_preprocessing else 'Torch_reference',
                all_selected_native_timestamps_retained=True))
