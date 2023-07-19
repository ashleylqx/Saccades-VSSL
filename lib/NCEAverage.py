import torch
from torch import nn
from .alias_multinomial import AliasMethod
import math
import numpy as np
import faiss
import time
from .normalize import Normalize
from random import sample
eps = 1e-7

class NCEAverage(nn.Module):
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, f_Num=3,
                 proj_dim=512):
        super(NCEAverage, self).__init__()
        self.nLem = outputSize * f_Num
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K
        self.f_Num = f_Num
        self.proj_dim = proj_dim

        self.register_buffer('params', torch.tensor([K, T, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(self.nLem, inputSize).mul_(2 * stdv).add_(-stdv))

        self.l2norm = Normalize(2)

        self.feature_proj = nn.Sequential(
            nn.Linear(inputSize, self.proj_dim),
        ) # NORMAL

        for m in self.feature_proj.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, feat, y, idx=None, mode='train', return_state=False, momentum=0.5):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z = self.params[2].item()

        batchSize = feat.size(0)

        feat_proj = self.feature_proj(feat)
        feat_proj = self.l2norm(feat_proj)
        if mode=='eval':
            return feat_proj

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
            idx = idx.to(feat.device)
        # sample
        weight = torch.index_select(self.memory, 0, idx.view(-1)).detach()
        weight_proj = self.feature_proj(weight)
        weight_proj = self.l2norm(weight_proj)
        weight_proj = weight_proj.view(batchSize, K + 1, self.proj_dim)
        out = torch.bmm(weight_proj, feat_proj.view(batchSize, self.proj_dim, 1))

        out = torch.div(out, T)
        out = out.contiguous()


        # # update memory
        with torch.no_grad():
            pos = torch.index_select(self.memory, 0, y.view(-1))
            pos.mul_(momentum)
            pos.add_(torch.mul(feat, 1 - momentum))
            updated = pos.clone()
            self.memory.index_copy_(0, y, updated)

        return out



DEFAULT_KMEANS_SEED = 1234
class NCEAverage_pcl(nn.Module):
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, f_Num=5, proj_dim=512):
        super(NCEAverage_pcl, self).__init__()
        self.nLem = outputSize * f_Num
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K
        self.f_Num = f_Num
        self.proj_dim = proj_dim
        self.cluster_result = None

        self.register_buffer('params', torch.tensor([K, T, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(self.nLem, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_curr', torch.rand(self.nLem, inputSize).mul_(2 * stdv).add_(-stdv))

        self.l2norm = Normalize(2)

        self.feature_proj = nn.Sequential(
            nn.Linear(inputSize, self.proj_dim),
        )  # NORMAL

        for m in self.feature_proj.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, feat, y, idx=None, mode='train'):  # y should contain info of vidx and fidx
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z = self.params[2].item()

        momentum = self.params[3].item()
        batchSize = feat.size(0)

        feat_proj = self.feature_proj(feat)
        feat_proj = self.l2norm(feat_proj)
        if mode == 'eval':
            return feat_proj

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize,
                                                                       -1)  # idx contains nce_k noise vectors
            idx.select(1, 0).copy_(y.data)  # oh, not y.size=(bs*(nce_k+1), 1), but copy y to the first slice of idx
            idx = idx.to(feat.device)
        # sample
        weight = torch.index_select(self.memory, 0, idx.view(-1)).detach()  # index_select(tensor, dim, indices)
        weight_proj = self.feature_proj(weight)
        weight_proj = self.l2norm(weight_proj)  # ****
        weight_proj = weight_proj.view(batchSize, K + 1, self.proj_dim)
        out = torch.bmm(weight_proj, feat_proj.view(batchSize, self.proj_dim, 1))

        out = torch.div(out, T)
        out = out.contiguous()

        # update memory
        with torch.no_grad():
            pos = torch.index_select(self.memory, 0, y.view(-1))
            pos.mul_(momentum)
            pos.add_(torch.mul(feat, 1 - momentum))
            # norm = pos.pow(2).sum(1, keepdim=True).pow(0.5)
            # updated = pos.div(norm)
            # self.memory.index_copy_(0, y, updated)
            updated = pos.clone()
            self.memory.index_copy_(0, y, updated)

        # prototypical contrast
        if self.cluster_result is not None:
            proto_labels = []
            proto_logits = []
            for n, (im2cluster, prototypes, density) in enumerate(
                    zip(self.cluster_result['im2cluster'], self.cluster_result['centroids'], self.cluster_result['density'])):
                im2cluster = im2cluster.to(feat.device)
                prototypes = prototypes.to(feat.device)
                density = density.to(feat.device)

                # get positive prototypes
                pos_proto_id = im2cluster[y]
                pos_prototypes = prototypes[pos_proto_id]

                # sample negative prototypes
                all_proto_id = [i for i in range(im2cluster.max() + 1)]
                neg_proto_id = set(all_proto_id) - set(pos_proto_id.tolist())
                if self.K < len(neg_proto_id):
                    neg_proto_id = sample(neg_proto_id, self.K)  # sample negative prototypes
                else:
                    # neg_proto_id = list(neg_proto_id)
                    neg_proto_id = sample(neg_proto_id, im2cluster.max() + 1 - pos_proto_id.size(0))  # sample negative prototypes
                neg_prototypes = prototypes[neg_proto_id]

                proto_selected = torch.cat([pos_prototypes, neg_prototypes], dim=0)

                # *** whether to project prototypes
                with torch.no_grad():
                    # proto_selected = self.feature_proj(proto_selected)
                    proto_selected = self.l2norm(proto_selected)

                # compute prototypical logits
                logits_proto = torch.mm(feat_proj, proto_selected.t())

                # targets for prototype assignment
                labels_proto = torch.linspace(0, batchSize - 1, steps=batchSize).long().cuda()

                # scaling temperatures for the selected prototypes
                temp_proto = density[torch.cat([pos_proto_id, torch.LongTensor(neg_proto_id).cuda()], dim=0)]
                logits_proto /= temp_proto

                proto_labels.append(labels_proto)
                proto_logits.append(logits_proto)
            return out, proto_logits, proto_labels
        else:
            return out, None, None

    def update_clust(self, num_clusters, device=0, verbose=False):
        print('Clustering memory into %s clusters ...' % num_clusters) # a list of clusters

        with torch.no_grad():
            # *** whether to norm memory slots before clustering
            # data = self.memory_curr[torch.norm(self.memory_curr,dim=1)>1.5]
            # data = data.cpu().numpy()
            data = self.memory_curr.cpu().numpy()

            start = time.time()
            cluster_result = self.run_kmeans(data.astype(np.float32), num_clusters, device, verbose=verbose)
            print('Time Elapsed: {:2.2f} seconds'.format(time.time() - start))

            self.cluster_result = cluster_result
        # return cluster_result

    def run_kmeans(self, x, num_clusters, device, verbose=False):
        """
        Args:
            x: data to be clustered
        """

        results = {'im2cluster': [], 'centroids': [], 'density': []}

        for seed, num_cluster in enumerate(num_clusters):
            # intialize faiss clustering parameters
            d = x.shape[1]
            k = int(num_cluster)
            clus = faiss.Clustering(d, k)
            clus.verbose = verbose
            clus.niter = 20
            clus.nredo = 5
            clus.seed = seed+DEFAULT_KMEANS_SEED
            clus.max_points_per_centroid = 1000
            clus.min_points_per_centroid = 10

            res = faiss.StandardGpuResources()
            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = False
            cfg.device = device
            index = faiss.GpuIndexFlatL2(res, d, cfg)

            clus.train(x, index)

            D, I = index.search(x, 1)  # for each sample, find cluster distance and assignments
            im2cluster = [int(n[0]) for n in I]

            # get cluster centroids
            centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

            # sample-to-centroid distances for each cluster
            Dcluster = [[] for c in range(k)]
            for im, i in enumerate(im2cluster):
                Dcluster[i].append(D[im][0])

            # concentration estimation (phi)
            density = np.zeros(k)
            for i, dist in enumerate(Dcluster):
                if len(dist) > 1:
                    d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                    density[i] = d

            # if cluster only has one point, use the max to estimate its concentration
            dmax = density.max()
            for i, dist in enumerate(Dcluster):
                if len(dist) <= 1:
                    density[i] = dmax

            density = density.clip(np.percentile(density, 10),
                                   np.percentile(density, 90))  # clamp extreme values for stability
            density = self.params[1].item() * density / density.mean()  # scale the mean to temperature

            # convert to cuda Tensors for broadcast
            centroids = torch.Tensor(centroids)

            im2cluster = torch.LongTensor(im2cluster)
            density = torch.Tensor(density)

            results['centroids'].append(centroids)
            results['density'].append(density)
            results['im2cluster'].append(im2cluster)

        return results

