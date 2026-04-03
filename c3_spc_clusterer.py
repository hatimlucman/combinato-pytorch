"""
C3 - SPCClusterer
==================
Runs Super-Paramagnetic Clustering (SPC) on feature vectors.

Wraps cluster_features.py exactly:
    - writes features to disk as text
    - writes a .run config file
    - calls SPC binary via subprocess
    - reads back .dg_01 (tree) and .dg_01.lab (labels) files

This module cannot be GPU-accelerated since SPC is an external C binary.
It is wrapped as nn.Module to maintain consistent pipeline interface.

Input:
    features: (N, 10) tensor — selected features from C2
    folder:   str — working directory for temp files
    name:     str — base name for temp files
    seed:     float — random seed for reproducibility

Output:
    clu:  numpy array (T, N+2) — cluster labels at each temperature
    tree: numpy array (T, ...)  — temperature tree
"""

import os
import time
import subprocess
import numpy as np
import torch
import torch.nn as nn

options = {
    'ClusterPath': r'C:\Users\hp\Desktop\combinato\combinato\combinato\cluster',
    'TempStep': 0.01,
    'ShowSPCOutput': False
}

# mirrors cluster_features.py constants
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


class SPCClusterer(nn.Module):
    """
    C3: Wrapper around SPC external binary.

    Mirrors cluster_features() from cluster_features.py exactly.
    Writes temp files, runs SPC, reads results back.

    Args:
        cluster_path (str): path to SPC binary. From options['ClusterPath'].
        temp_step (float): temperature step. From options['TempStep'].
    """

    def __init__(self, cluster_path=None, temp_step=None):
        super().__init__()
        self.cluster_path = cluster_path or options.get('ClusterPath', 'cluster')
        self.temp_step    = temp_step    or options.get('TempStep', 0.01)

    def forward(self, features, folder, name, seed):
        """
        Run SPC on feature matrix.

        Args:
            features: (N, K) tensor or numpy array
            folder:   str — working directory
            name:     str — base filename
            seed:     float — random seed

        Returns:
            clu:  numpy array — cluster labels per temperature
            tree: numpy array — temperature tree
        """
        if isinstance(features, torch.Tensor):
            features_np = features.cpu().numpy()
        else:
            features_np = features

        if not os.path.isdir(folder):
            os.mkdir(folder)

        cleanname = os.path.join(folder, name)

        if DO_CLEAN:
            _cleanup(cleanname, EXT_CL)

        # write feature data to disk
        data_fname    = name + '_tmp_data'
        datasavename  = os.path.join(folder, data_fname)
        np.savetxt(datasavename, features_np, newline='\n', fmt='%f')

        # write SPC config file
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

        # run SPC binary
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

        # read results
        clu, tree = self._read_results(folder, name)
        return clu, tree

    def _read_results(self, folder, name):
        tree_fname = os.path.join(folder, name + '.dg_01')
        clu_fname  = os.path.join(folder, name + '.dg_01.lab')
        tree = np.loadtxt(tree_fname)
        clu  = np.loadtxt(clu_fname)
        return clu, tree


if __name__ == '__main__':
    print("C3 SPCClusterer defined.")
    print("Requires SPC binary — test via test_c1_to_c4.py after C4 is built.")
