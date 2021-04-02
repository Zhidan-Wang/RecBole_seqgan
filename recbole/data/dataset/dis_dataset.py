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

class DisDataset(Dataset):
    def __init__(self, config, data_file, saved_dataset=None):
        super().__init__(config, saved_dataset=saved_dataset)
        print('###########', self.user_num)
        self.g_file = data_file
        g_list = self.read_file(self.g_file)
        gene = self.get_gene(g_list)

        #sample_index = np.random.choice(999611, 50000)
        #self.inter_feat = self.inter_feat[sample_index]
        self.inter_feat = cat_interactions([self.inter_feat, gene])
        #print("inter feat ############", self.inter_feat)
        self.prepare_data_augmentation()

        #print("!!!!!!!!!!!!!!!!!!!!", self.item_list_index)
        print("target!!!!!!!!!!!!!!", len(self.uid_list))
        pos_tot = 0
        for i in range(len(self.uid_list)):
            if self.label[i] == 1:
                pos_tot += 1

        print("label!!!!!!!!!!!!!!!", pos_tot)

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
                res = res.append([{'user_id':int(user_num + i) / 1.0,  'item_id':int(seq[j]) / 1.0,  'timestamp':int(j) / 1.0}], ignore_index=True)
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
        uid_list, item_list_index, target_index, item_list_length, label = [], [], [], [], []
        seq_start = 0
        real_cnt = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if last_uid != uid:
                #uid_list.append(uid)
                #item_list_index.append([i])
                #item_list_length.append(1)
                #target_index.append(i)

                last_uid = uid
                if last_uid >= self.user_num:
                    uid_list.append(uid)
                    item_list_index.append([i])
                    item_list_length.append(1)
                    target_index.append(i)

                    label.append(0)
                else:
                    real_cnt += 1
                    if real_cnt > 100000:
                        continue

                    uid_list.append(uid)
                    item_list_index.append([i])
                    item_list_length.append(1)
                    target_index.append(i)
                    label.append(1)
                seq_start = i
            else:
                if i - seq_start + 1 > max_item_list_len:
                    seq_start += 1
                #uid_list.append(uid)
                #item_list_index.append(slice(seq_start, i + 1))
                #target_index.append(i)
                #item_list_length.append(i + 1 - seq_start)

                if last_uid >= self.user_num:
                    uid_list.append(uid)
                    item_list_index.append(slice(seq_start, i + 1))
                    target_index.append(i)
                    item_list_length.append(i + 1 - seq_start)
                    label.append(0)
                else:
                    real_cnt += 1
                    if real_cnt > 100000:
                        continue
                    uid_list.append(uid)
                    item_list_index.append(slice(seq_start, i + 1))
                    target_index.append(i)
                    item_list_length.append(i + 1 - seq_start)
                    label.append(1)

        self.uid_list = np.array(uid_list)
        self.item_list_index = np.array(item_list_index)
        self.target_index = np.array(target_index)
        self.item_list_length = np.array(item_list_length, dtype=(np.int64))
        self.label = np.array(label, dtype=(np.int64))

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

    def leave_one_out(self, group_by, leave_one_num=1):
        self.logger.debug(f'Leave one out, group_by=[{group_by}], leave_one_num=[{leave_one_num}].')
        if group_by is None:
            raise ValueError('Leave one out strategy require a group field.')
        if group_by != self.uid_field:
            raise ValueError('Sequential models require group by user.')
        self.prepare_data_augmentation()
        grouped_index = self._grouped_index(self.uid_list)
        next_index = self._split_index_by_leave_one_out(grouped_index, leave_one_num)

        self._drop_unused_col()
        next_ds = []
        for index in next_index:
            ds = copy.copy(self)
            for field in ['uid_list', 'item_list_index', 'target_index', 'item_list_length','label']:
                setattr(ds, field, np.array(getattr(ds, field)[index]))
            next_ds.append(ds)
        return next_ds

    def split_by_ratio(self, ratios, group_by=None):
        self.logger.debug(f'split by ratios [{ratios}], group_by=[{group_by}]')
        tot_ratio = sum(ratios)
        ratios = [_ / tot_ratio for _ in ratios]

        self.prepare_data_augmentation()

        if group_by is None:
            tot_cnt = self.__len__()
            split_ids = self._calcu_split_ids(tot=tot_cnt, ratios=ratios)
            next_index = [range(start, end) for start, end in zip([0] + split_ids, split_ids + [tot_cnt])]
        else:
            grouped_inter_feat_index = self._grouped_index(self.uid_list)
            next_index = [[] for _ in range(len(ratios))] # 0,1,2
            for grouped_index in grouped_inter_feat_index:
                tot_cnt = len(grouped_index)
                split_ids = self._calcu_split_ids(tot=tot_cnt, ratios=ratios)
                for index, start, end in zip(next_index, [0] + split_ids, split_ids + [tot_cnt]):
                    index.extend(grouped_index[start:end])

        self._drop_unused_col()
        next_ds = []
        for index in next_index:
            ds = copy.copy(self)
            for field in ['uid_list', 'item_list_index', 'target_index', 'item_list_length','label']:
                setattr(ds, field, np.array(getattr(ds, field)[index]))
            next_ds.append(ds)
        return next_ds

    def build(self, eval_setting):
        ordering_args = eval_setting.ordering_args
        if ordering_args['strategy'] == 'shuffle':
            raise ValueError('Ordering strategy `shuffle` is not supported in sequential models.')
        elif ordering_args['strategy'] == 'by':
            if ordering_args['field'] != self.time_field:
                raise ValueError('Sequential models require `TO` (time ordering) strategy.')
            if ordering_args['ascending'] is not True:
                raise ValueError('Sequential models require `time_field` to sort in ascending order.')

        group_field = eval_setting.group_field

        split_args = eval_setting.split_args
        if split_args['strategy'] == 'loo':
            return self.leave_one_out(group_by=group_field, leave_one_num=split_args['leave_one_num'])
        elif split_args['strategy'] == 'by_ratio':
            return self.split_by_ratio(split_args['ratios'], group_by=group_field)
        else:
            ValueError('Sequential models require `loo` (leave one out) split strategy.')
