from collections import defaultdict
from torch.utils.data import Sampler


class SentenceBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last=False):
        super(SentenceBatchSampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.trigger_label_map = defaultdict(lambda: defaultdict(set))
        self.label_sent_map = defaultdict(set)
        self.sent_label_map = defaultdict(set)
        self.has_event_ids = set()
        self.cluster_size = 4
        for i, (data, indexes, labels) in enumerate(
                zip(self.data_source, self.data_source.trigger_indexes, self.data_source.trigger_labels)):
            sentence, _, _ = data
            if len(indexes) > 0:
                self.has_event_ids.add(i)
                for label, index in zip(labels, indexes):
                    trigger = sentence[index]
                    self.trigger_label_map[trigger][label].add(i)
                    self.label_sent_map[label].add(i)
                    self.sent_label_map[i].add(label)
        self.no_event_ids = set(range(len(self.data_source))) - self.has_event_ids

    def __iter__(self):
        import random
        random.shuffle(self.trigger_label_map)
        half_batch = []
        for i, trigger in enumerate(self.trigger_label_map):
            if i % self.cluster_size == 0:
                half_batch = []
            for sent_ids in self.trigger_label_map[trigger].values():
                half_batch.append(random.sample(sent_ids, 1)[0])
            if i % self.cluster_size != 3:
                continue
            half_batch_size = self.batch_size // 2
            if len(half_batch) > half_batch_size:
                half_batch = random.sample(half_batch, half_batch_size)
            elif len(half_batch) < half_batch_size:
                half_pad_num = (half_batch_size - len(half_batch)) // 2
                half_batch.extend(random.sample(self.has_event_ids - set(half_batch), half_batch_size - len(half_batch) - half_pad_num))
                half_batch.extend(random.sample(self.no_event_ids - set(half_batch), half_pad_num))
            assert len(half_batch) == self.batch_size // 2
            random.shuffle(half_batch)
            include_set = set(half_batch)
            batch = []
            for sent_id in half_batch:
                batch.append(sent_id)
                if sent_id in self.no_event_ids:
                    batch.append(random.sample(self.no_event_ids, 1)[0])
                    continue
                candidates = set()
                for label in self.sent_label_map[sent_id]:
                    candidates |= self.label_sent_map[label]
                candidates -= include_set
                if len(candidates) == 0:
                    pair_sent_id = sent_id
                else:
                    pair_sent_id = random.sample(candidates, 1)[0]
                include_set.add(pair_sent_id)
                batch.append(pair_sent_id)
            assert len(batch) == self.batch_size
            # for sent_id in batch:
            #     print(sent_id)
            #     print("label:", self.data_source.trigger_labels[sent_id])
            #     print("trigger:", self.data_source.sentences[sent_id][self.data_source.trigger_indexes[sent_id]])
            # input()
            yield batch

    def __len__(self):
        return len(self.trigger_label_map) // self.cluster_size
