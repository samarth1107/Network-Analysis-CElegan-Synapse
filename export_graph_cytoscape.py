import os
import re

import numpy as np
import scipy as sc
import networkx as nx

def nclass(n):
    if n in (
        'AVG', 'DVC', 'PVR', 'PVT', 'RIH', 'RIR', 'DVA', 'AQR', 'AVM', 'PQR',
        'PVM', 'DVB', 'PDA', 'PDB', 'ALA', 'AVL', 'RID', 'RIS',
        'I3', 'I4', 'I5', 'I5', 'M1', 'M4', 'M5', 'MI'
    ):
        return n
    if len(n) == 4 and n[-1] in 'LR' and n[:3] in (
        'ADA', 'ADE', 'ADF', 'ADL', 'AFD', 'AIA', 'AIB', 'AIM', 'AIN', 'AIY',
        'AIZ', 'ALM', 'ALN', 'ASE', 'ASG', 'ASH', 'ASI', 'ASJ', 'ASK', 'AUA',
        'AVA', 'AVB', 'AVD', 'AVE', 'AVF', 'AVH', 'AVJ', 'AVK', 'AWA', 'AWB',
        'AWC', 'BAG', 'BDU', 'CAN', 'FLP', 'GLR', 'HSN', 'IL1', 'IL2', 'LUA',
        'OLL', 'PDE', 'PHA', 'PHB', 'PHC', 'PLM', 'PLN', 'PVC', 'PVD', 'PVN',
        'PVP', 'PVQ', 'PVW', 'RIA', 'RIB', 'RIC', 'RIF', 'RIG', 'RIM', 'RIP',
        'RIV', 'RMD', 'RMF', 'RMG', 'RMH', 'SDQ', 'URB', 'URX'
    ):
        return n[:3]
    if len(n) == 5 and n[-2:] in ('DL', 'DR', 'VL', 'VR') and n[:3] in (
        'CEP', 'GLR', 'IL1', 'IL2', 'OLQ', 'RMD', 'SAA', 'SIA', 'SIB', 'SMB',
        'SMD', 'URA', 'URY'
    ):
        return n[:3]
    if len(n) == 8 and re.match('BWM-[DV][LR]0[0-8]', n):
        return 'BWM' + n[-2:]
    if n in (
        'RMED', 'RMEL', 'RMER', 'RMEV', 'SABD', 'SABVL', 'SABVR',
    ):
        return n[:3]
    if n in (
        'CEPshDL', 'CEPshDR', 'CEPshVL', 'CEPshVR'
    ):
        return n[:5]
    if n[:2] in ('AS', 'VB', 'VA', 'VD') and n[2:] in map(str, range(12)):
        return n[:2] + 'n'
    if n in ('VA12', 'VD12', 'VD13'):
        return n[:2] + 'n'
    if re.match('^(DA[1-9])|(DB[1-7])|(DD[1-6])|(VC[1-6])$', n):
        return n[:2] + 'n'
    return n

def ntype(n):
    n = nclass(n)

    if n in (
        'ADF', 'ADL', 'AFD', 'ALM', 'ALN', 'AQR', 'ASE', 'ASG', 'ASH', 'ASI',
        'ASJ', 'ASK', 'AUA', 'AVM', 'AWA', 'AWB', 'AWC', 'BAG', 'DVA', 'FLP',
        'IL2', 'OLL', 'OLQ', 'PHA', 'PHB', 'PHC', 'PLM', 'PLN', 'PQR', 'PVD',
        'PVM', 'SAA', 'SDQ', 'URB', 'URX', 'URY'
    ):
        return 'sensory'
    if n in (
        'ADA', 'AIA', 'AIB', 'AIN', 'AIY', 'AIZ', 'AVA', 'AVB', 'AVD', 'AVE',
        'AVG', 'BDU', 'LUA', 'PVC', 'PVP', 'PVR', 'PVT', 'PVW',
        'RIA', 'RIB', 'RIF', 'RIG', 'RIH', 'RIM', 'RIR', 'RIP', 'AVJ',
    ):
        return 'inter'
    if n in (
        'ASn', 'DAn', 'DBn', 'DDn', 'DVB', 'IL1', 'PDA', 'PDB', 'RIV', 'RMD',
        'RME', 'RMF', 'RMH', 'SAB', 'SIA', 'SIB', 'SMB', 'SMD', 'URA', 'VAn',
        'VBn', 'VCn', 'VDn',
    ):
        return 'motor'
    if n in (
        'ADE', 'AIM', 'ALA', 'AVF', 'AVH', 'AVK', 'AVL', 'CEP', 'HSN',
        'PDE', 'PVQ', 'PVN', 'RIC', 'RID', 'RIS', 'RMG', 'DVC',
    ):
        return 'modulatory'
    if n in (
        'BWM01', 'BWM02', 'BWM03', 'BWM04', 'BWM05', 'BWM06', 'BWM07', 'BWM08'
    ):
        return 'muscle'
    if n in ('CAN', 'CEPsh', 'GLR', 'excgl', 'hyp'):
        return 'other'
    print(n, 'is not a valid neuron')
    return 'nonvalid'

def is_neuron(n):
    n = nclass(n)
    return ntype(n) not in ('muscle', 'other')

def is_postemb(n):
    n = nclass(n)
    return n in (
        'ALN', 'AQR', 'ASn', 'AVF', 'AVM', 'DVB', 'HSN', 'PDA', 'PDB', 'PDE',
        'PHC', 'PLN', 'PQR', 'PVD', 'PVM', 'PVN', 'PVW', 'RMF', 'RMH', 'SDQ',
        'VAn', 'VBn', 'VCn', 'VDn'
    )

def export_graphs_for_cytoscape(save_to, connection_data, edge_classifications):

    categories = {
        'increase': 'Developmental change',
        'decrease': 'Developmental change',
        'stable': 'Stable',
        'noise': 'Variable',
        'remainder': 'Variable'
    }

    classifications = edge_classifications

    G = connection_data.copy()
    G_normalized = G / G.sum() * G.sum().mean()

    graph = nx.DiGraph()

    for (pre, post), syns in G.iterrows():
        if pre == post:
            continue
        if is_postemb(pre) or is_postemb(post):
            classification = 'Postemb'
        else:
            classification = categories[classifications[(pre, post)]]

        edge_properties = {
            'weight': max(syns), 'classification': classification,
            'weight_normalized': G_normalized.loc[(pre, post)].max(),
            'transparency_stable': max(syns) if classification == 'Stable' else 0,
            'transparency_variable': max(syns) if classification == 'Variable' else 0,
            'transparency_changing': max(syns) if classification == 'Developmental change' else 0,
            'nomodule': int(is_postemb(pre) or is_postemb(post) or ntype(post) == 'other')
        }

        for i, d in enumerate(['Dataset1','Dataset2','Dataset3','Dataset4','Dataset5','Dataset6','Dataset7','Dataset8']):
            edge_properties[d] = syns[i]

        graph.add_edge(pre, post, **edge_properties)

    for n in ('CANL', 'CANR', 'excgl'):
        if graph.has_node(n):
            graph.remove_node(n)

    for n in graph.nodes():
        typ = ntype(n)
        graph.nodes[n]['type'] = ntype(n)
        graph.nodes[n]['celltype'] = 'neuron' if is_neuron(n) else typ
        graph.nodes[n]['is_postemb'] = int(is_postemb(n))
        graph.nodes[n]['is_neuron'] = int(is_neuron(n))
        graph.nodes[n]['no_module'] = int(is_postemb(n) or typ == 'other')
        graph.nodes[n]['hidden_l2-l3'] = int(n in ('HSNL', 'HSNR', 'PVNL', 'PVNR', 'PLNL', 'PLNR'))
        graph.nodes[n]['hidden_latel1'] = int(n in ('HSNL', 'HSNR', 'PVNL', 'PVNR', 'PLNL', 'PLNR', 'AVFL', 'AVFR', 'AVM', 'RMFL' 'RMFR'))
        graph.nodes[n]['hidden_l1'] = int(is_postemb(n))

    graphs = {'combined': graph}
    for d in ['Dataset1','Dataset2','Dataset3','Dataset4','Dataset5','Dataset6','Dataset7','Dataset8']:
        graphs[d] = graph.copy()
        for (pre, post) in list(graphs[d].edges()):
            weight = graphs[d][pre][post][d]
            if weight == 0:
                graphs[d].remove_edge(pre, post)
            else:
                graphs[d][pre][post]['weight'] = weight
        graphs[d].remove_nodes_from(list(nx.isolates(graphs[d])))

    for dataset, G in graphs.items():
        # number of nodes
        nodelist = list(G.nodes())
        A = np.array(nx.adjacency_matrix(G, nodelist=nodelist, weight='weight_normalized').todense()).astype(float)

        # symmetrize the adjacency matrix
        c = (A + np.transpose(A))/2.0

        # degree matrix
        d = np.diag(np.sum(c, axis=0))
        df = sc.linalg.fractional_matrix_power(d, -0.5)

        # Laplacian matrix
        l = d - c

        # compute the vertical coordinates
        b = np.sum(c * np.sign(A - np.transpose(A)), 1)
        z = np.matmul(np.linalg.pinv(l), b)

        # degree-normalized graph Laplacian
        q = np.matmul(np.matmul(df, l), df)

        # coordinates in plane are eigenvectors of degree-normalized graph Laplacian
        _, vx = np.linalg.eig(q)
        x = np.matmul(df, vx[:, 1])
        y = np.matmul(df, vx[:, 2])

        for n in graph.nodes():
            key = '' if dataset == 'combined' else dataset + '_'
            if n not in nodelist:
                graph.nodes[n][key+'x'] = 0.0
                graph.nodes[n][key+'y'] = 0.0
                graph.nodes[n][key+'z'] = 0.0
                graph.nodes[n][key+'zorder'] = 0.0
                graph.nodes[n][key+'visibility'] = 0
                continue

            i = nodelist.index(n)
            graph.nodes[n][key+'x'] = x[i] * 11500.0
            graph.nodes[n][key+'y'] = y[i] * 11500.0
            graph.nodes[n][key+'z'] = z[i] * -150.0
            graph.nodes[n][key+'zorder'] = -z[i] if ntype(n) != 'other' else -z[i]-100000  # put glia in back
            graph.nodes[n][key+'visibility'] = 1

    print('x range:', min(x), max(x))
    print('y range:', min(z), max(z))

    for pre, post in graph.edges():
        graph[pre][post]['bend'] = int(graph.nodes[pre]['z'] < graph.nodes[post]['z'])

    fpath = os.path.join(save_to, 'graphs_for_cytoscape.graphml')
    nx.write_graphml(graph, fpath)
    print(f'Saved to `{fpath}`')
