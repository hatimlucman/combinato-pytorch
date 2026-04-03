"""
C3 - SPCClusterer
==================
Runs Super-Paramagnetic Clustering (SPC) on feature vectors.
"""

import os
import time
import subprocess
import numpy as np
import torch
from block import Block

options = {
    'ClusterPath': r'C:\Users\hp\Downloads\combinato\spc',
    'TempStep': 0.01,
    'ShowSPCOutput': False
}

DO_CLEAN  = True
DO_RUN    = True
DO_TIMING = True
EXT_CL    = ('.dg_01', '.dg_01.lab')
EXT_TMP   = ('.mag', '.mst11.edges', '.param', '_tmp_data', '_cluster.run')


def _cleanup(base, ext):
    for e in ext:
        name = base + e
        if os.path.exists(name):
            os.remove(name)


class SPCClusterer(Block):
    def __init__(self, cluster_path=None, temp_step=None):
        super().__init__()
        self.cluster_path = cluster_path or options.get('ClusterPath', 'cluster')
        self.temp_step    = temp_step    or options.get('TempStep', 0.01)

    def forward(self, features, folder, name, seed):
        print("SPC seed used:", seed)
        if isinstance(features, torch.Tensor):
            features_np = features.cpu().numpy()
        else:
            features_np = features

        if not os.path.isdir(folder):
            os.mkdir(folder)

        cleanname = os.path.join(folder, name)

        if DO_CLEAN:
            _cleanup(cleanname, EXT_CL)

        data_fname   = name + '_tmp_data'
        datasavename = os.path.join(folder, data_fname)
        np.savetxt(datasavename, features_np, newline='\n', fmt='%f')

        run_fname_base = name + '_cluster.run'
        run_fname      = os.path.join(folder, run_fname_base)
        with open(run_fname, 'w') as fid:
            fid.write('NumberOfPoints: %i\n'  % features_np.shape[0])
            fid.write('DataFile: %s\n'        % data_fname)
            fid.write('OutFile: %s\n'         % name)
            fid.write('Dimensions: %s\n'      % features_np.shape[1])
            fid.write('MinTemp: 0\n')
            fid.write('MaxTemp: 0.201\n')
            fid.write('TempStep: %f\n'        % self.temp_step)
            fid.write('SWCycles: 100\n')
            fid.write('KNearestNeighbours: 11\n')
            fid.write('MSTree|\n')
            fid.write('DirectedGrowth|\n')
            fid.write('SaveSuscept|\n')
            fid.write('WriteLables|\n')
            fid.write('WriteCorFile~\n')
            fid.write('ForceRandomSeed: %f\n' % seed)

        out = None if options.get('ShowSPCOutput', False) else subprocess.PIPE
        if DO_RUN:
            t0  = time.time()
            ret = subprocess.call((self.cluster_path, run_fname_base),
                                  stdout=out, cwd=folder)
            dt  = time.time() - t0
            if ret:
                raise RuntimeError('SPC failed for: ' + name)
            if DO_TIMING:
                log = os.path.join(folder, 'cluster_log.txt')
                with open(log, 'a') as f:
                    f.write('clustered {} spikes in {:.4f}s\n'.format(
                        features_np.shape[0], dt))

        if DO_CLEAN:
            _cleanup(cleanname, EXT_TMP)

        clu, tree = self._read_results(folder, name)
        return clu, tree

    def _read_results(self, folder, name):
        tree_fname = os.path.join(folder, name + '.dg_01')
        clu_fname  = os.path.join(folder, name + '.dg_01.lab')
        tree = np.loadtxt(tree_fname)
        clu  = np.loadtxt(clu_fname)
        return clu, tree
