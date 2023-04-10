import authmecv as acv
import numpy as np
import pandas as pd

import faiss


def exhaustive_search_benchmark(k: int, gt: list, param: str, output_csv_path: str):
    # exhasutive search
    exhaustive_index_timer = acv.Timer()
    with exhaustive_index_timer:
        index = faiss.index_factory(d, param, faiss.METRIC_INNER_PRODUCT)
        index.ntotal
        faiss.normalize_L2(xb)
        index.add(xb)
        faiss.normalize_L2(xq)
    exhaustive_search_timer = acv.Timer()
    with exhaustive_search_timer:
        res_distance, res_index = index.search(xq, k)  # return top-1 for every query vector
    # print('Index by FAISS:{}'.format(res_index))
    # print('Distance by FAISS:{}'.format(res_distance))
    # print(f"Top-1 Sim of query vector {features['path'][0]} is {features['path'][res_index][0][0]}")
    result_df = pd.DataFrame(columns=['query_res', 'gt', 'cos_sim', 'label'])
    for i, query_result_index in acv.Tqdm(enumerate(res_index)):
        # If query result is itself, then label is 1, otherwise 0
        if query_result_index[0] == gt[i]:
            new_row = pd.DataFrame({'query_res': features['path'][query_result_index[0]], 'gt': features['path'][gt[i]], 'cos_sim': res_distance[i][0].round(3), 'label': 1.0}, index=[0])
            result_df = pd.concat([new_row, result_df.loc[:]]).reset_index(drop=True)
        else:
            new_row = pd.DataFrame({'query_res': features['path'][query_result_index[0]], 'gt': features['path'][gt[i]], 'cos_sim': res_distance[i][0].round(3), 'label': 0.0}, index=[0])
            result_df = pd.concat([new_row, result_df.loc[:]]).reset_index(drop=True)
    result_df.to_csv(output_csv_path, index=False)


if __name__ == '__main__':
    features = dict(np.load('/disks/pcie_2t_1/authmeqa/celeba_11_verification/ori_face_features.npz'))
    # print(features.keys()) # ['path', 'features', 'feature_version']
    # print(features['features'].shape) # (196766, 1024)
    # print(features['path'][0]) # CelebA_1/0001435.json
    xb = features['features']
    xq = features['features'][:]
    d = features['features'].shape[1]
    gt = [i for i in range(len(features['path']))]
    k = 1

    output_csv_path = 'exhaustive_result.csv'
    param = 'Flat'
    exhaustive_search_benchmark(k, gt, param, output_csv_path)
    result = pd.read_csv(output_csv_path)
    scores = result['cos_sim'].to_list()
    labels = result['label'].to_list()
    print('Accuracy:', len(result[result.label == 1.0]) / len(result))
    print('Number of Wrong:', len(result[result.label == 0.0]))
