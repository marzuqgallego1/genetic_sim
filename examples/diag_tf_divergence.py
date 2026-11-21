import sys, numpy as np
sys.path.insert(0, "/home/marco_aurelio_lorenzo/live_sim")
from evosim import codon_architecture, StabilizingSelection
from evosim.speciation import Radiation, count_dmis, hybrid_fitness, isolation_index
from evosim.stats import consensus
rng = np.random.default_rng(5)
arch = codon_architecture(rng, n_enzyme_genes=10, n_trait_genes=4, n_traits=1, n_tf=8, tf_targets=1, tf_mu=6.0, tf_target_roles=("enzyme",),
        codons_per_gene=40, promoter_len=20, intergenic_len=40, mutation_rate=5e-5, promoter_effect_sd=0.1)
gm=arch.gmap; env=StabilizingSelection([0.0],1.0)
from evosim import Population
import os
if os.path.exists("burn_tf1.npy"): g0=np.load("burn_tf1.npy")
else:
    p=Population(arch,200,rng,environment=env,selfing=False)
    for k in range(20):
        p.run(1000); print("burn",p.generation,round(float(np.exp(p.evaluate()[0]).mean()),3),flush=True)
    g0=p.genomes; np.save("burn_tf1.npy",g0)
rad=Radiation(arch,200,rng,env,genomes=g0); a,b=rad.split("L0")
def report():
    ga,gb=rad.lineages[a].pop.genomes, rad.lineages[b].pop.genomes
    ca,cb=consensus(ga),consensus(gb)
    print("gen",rad.generation)
    for r in gm.regulations:
        rs=gm._recog_sites[r.tf].ravel()
        def E(tfg, bsg):
            g=np.broadcast_to(tfg,(1,2,len(tfg))).copy(); g[0][:, r.site]=bsg[r.site]
            return gm.binding_energy(g,r)[0,0,0]
        print(f" tf{r.tf}->g{r.target} imp={gm.genes[r.target].importance:.2f} K={gm.genes[r.target].K:.2f} "
              f"anc-E={E(arch.ancestral,arch.ancestral):.1f} A:E={E(ca,ca):.1f} B:E={E(cb,cb):.1f} TFa/BSb={E(ca,cb):.1f} TFb/BSa={E(cb,ca):.1f} "
              f"recogdiff={int((ca[rs]!=cb[rs]).sum())} bsdiff={int((ca[r.site]!=cb[r.site]).sum())}")
report()
for k in range(5):
    rad.run(2000); report()
    ga,gb=rad.lineages[a].pop.genomes, rad.lineages[b].pop.genomes
    hf=hybrid_fitness(arch,ga,gb,rng,env,n=1500); n,K,_=count_dmis(arch,ga,gb,tol=0.02)
    print('  strongDMI',n,'K',K,'RI_F1 %.3f RI_F2 %.3f'%(isolation_index(hf,'F1'),isolation_index(hf,'F2')),flush=True)
