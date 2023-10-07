from typing import Callable, List, Optional

import torch

from torch_geometric.data import HeteroData, InMemoryDataset, download_url


class UpperToBottom(InMemoryDataset):
    url = ('')

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        return ['upper_class_list.txt', 'bottom_item_list.txt', 'train.txt', 'test.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{name}', self.raw_dir)

    def process(self):
        import pandas as pd

        data = HeteroData()
        print(f"\x1b[32m\"상의류(스타일, 중분류, 성별) 클래스 ➡ 하의아이템 추천 datasets 구축. {data}\"\x1b[0m")

        # Process number of nodes for each node type:
        # 각 노드 유형에 대한 프로세스 노드 수
        node_types = ['upper', 'bottom']
        for path, node_type in zip(self.raw_paths, node_types):
            df = pd.read_csv(path, sep=' ', header=0)
            data[node_type].num_nodes = len(df)

        # Process edge information for training and testing:
        # 학습 및 테스트를 위한 엣지 정보를 처리합니다:
        attr_names = ['edge_index', 'edge_label_index']
        for path, attr_name in zip(self.raw_paths[2:], attr_names):
            rows, cols = [], []
            with open(path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')
                for dst in line[1:]:
                    rows.append(int(line[0]))
                    cols.append(int(dst))
            index = torch.tensor([rows, cols])

            data['upper', 'rates', 'bottom'][attr_name] = index
            if attr_name == 'edge_index':
                data['bottom', 'rated_by', 'upper'][attr_name] = index.flip([0])

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
