"""
recbole.data.dataset
##########################
"""
import copy, torch
import torch.nn.utils.rnn as rnn_utils
import numpy as np, pandas as pd
from recbole.data.dataset import Dataset
from recbole.data.interaction import Interaction, cat_interactions
from recbole.utils import FeatureSource, FeatureType

class GanDataset(Dataset):

    def __init__(self, config, data_file, saved_dataset=None):
        super().__init__(config, saved_dataset=saved_dataset)
        self.g_file = data_file
        g_list = self.read_file(self.g_file)
        self.inter_feat = self.get_gene(g_list)
        self.prepare_data_augmentation()

    def read_file(self, data_file):
        with open(data_file, 'r') as (f):
            lines = f.readlines()
        lis = []
        for line in lines:
            l = line.strip().split(' ')
            l = [int(s) for s in l]
            lis.append(l)
        return lis

    def get_gene(self, list):
        user_num = self.user_num
        res = pd.DataFrame(columns=('user_id', 'item_id', 'timestamp'))
        for i, seq in enumerate(list):
            for j in range(len(seq)):
                res = res.append(
                    [{'user_id': int(user_num + i) / 1.0, 'item_id': int(seq[j]) / 1.0, 'timestamp': int(j) / 1.0}],
                    ignore_index=True)
        #print("res!!!!!!!!!!!", res)
        res = self._dataframe_to_interaction(res)
        return res

    def _dataframe_to_interaction(self, data):
        """Convert :class:`pandas.DataFrame` to :class:`~recbole.data.interaction.Interaction`.

        Args:
            data (pandas.DataFrame): data to be converted.

        Returns:
            :class:`~recbole.data.interaction.Interaction`: Converted data.
        """
        new_data = {}
        for k in data:
            value = data[k].values
            ftype = self.field2type[k]
            if ftype == FeatureType.TOKEN:
                new_data[k] = torch.LongTensor(value)
            elif ftype == FeatureType.FLOAT:
                new_data[k] = torch.FloatTensor(value)
            elif ftype == FeatureType.TOKEN_SEQ:
                seq_data = [torch.LongTensor(d[:self.field2seqlen[k]]) for d in value]
                new_data[k] = rnn_utils.pad_sequence(seq_data, batch_first=True)
            elif ftype == FeatureType.FLOAT_SEQ:
                seq_data = [torch.FloatTensor(d[:self.field2seqlen[k]]) for d in value]
                new_data[k] = rnn_utils.pad_sequence(seq_data, batch_first=True)
        return Interaction(new_data)

    def prepare_data_augmentation(self):
        """Augmentation processing for sequential dataset.

        E.g., ``u1`` has purchase sequence ``<i1, i2, i3, i4>``,
        then after augmentation, we will generate three cases.

        ``u1, <i1> | i2``

        (Which means given user_id ``u1`` and item_seq ``<i1>``,
        we need to predict the next item ``i2``.)

        The other cases are below:

        ``u1, <i1, i2> | i3``

        ``u1, <i1, i2, i3> | i4``

        Note:
            Actually, we do not really generate these new item sequences.
            One user's item sequence is stored only once in memory.
            We store the index (slice) of each item sequence after augmentation,
            which saves memory and accelerates a lot.
        """
        self.logger.debug('prepare_data_augmentation')
        self._check_field('uid_field', 'time_field')
        max_item_list_len = self.config['MAX_ITEM_LIST_LENGTH']
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length, item_seq = [], [], [], [], []
        seq_start = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if last_uid != uid:

                uid_list.append(uid)
                item_list_index.append([])
                item_list_length.append(1)
                target_index.append(i)
                item_seq.append([i])

                last_uid = uid
                seq_start = i
            else:
                if i - seq_start + 1 > max_item_list_len:
                    seq_start += 1
                uid_list.append(uid)
                item_list_index.append(slice(seq_start, i))
                item_seq.append(slice(seq_start, i + 1))
                target_index.append(i)
                item_list_length.append(i + 1 - seq_start)

        self.uid_list = np.array(uid_list)
        self.item_list_index = np.array(item_list_index)
        self.target_index = np.array(target_index)
        self.item_seq = np.array(item_seq)
        self.item_list_length = np.array(item_list_length, dtype=(np.int64))

    def prepare_data_augmentation111(self):
        self._check_field('uid_field', 'time_field')
        max_item_list_len = self.config['MAX_ITEM_LIST_LENGTH']
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list = []
        item_list_dict = dict()
        item_list_index = []
        item_list_length = []
        item_list_len = []
        seq_start = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if last_uid != uid:
                uid_list.append(uid)
                last_uid = uid
                cnt = 0
                item_list_dict[last_uid] = []
                item_list_dict[last_uid].append(i)
            else:
                item_list_dict[last_uid].append(i)
                cnt += 1
        else:
            self.uid_list = np.array(uid_list)
            for id, uid in enumerate(self.uid_list):
                if len(item_list_dict[uid]) > max_item_list_len - 1:
                    item_list_index.append(item_list_dict[uid][:max_item_list_len - 1])
                else:
                    item_list_index.append(item_list_dict[uid])
                item_list_length.append(len(item_list_index[id]))
                item_list_len.append((id, len(item_list_index[id])))
            else:
                item_list_len = sorted(item_list_len, key=(lambda x: x[1]))
                index = [i for i, _ in item_list_len]
                print('!!!!!!!!!!!!', index)
                self.uid_list = [self.uid_list[id] for id in index]
                item_list_index = [item_list_index[id] for id in index]
                item_list_length = [item_list_length[id] for id in index]
                self.item_list_index = np.array(item_list_index)
                self.item_list_length = np.array(item_list_length, dtype=(np.int64))
                self.item_list_len = np.array(item_list_len)