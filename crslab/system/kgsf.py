# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2021/1/3
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import os

import torch
from loguru import logger

from crslab.evaluator.metrics.base import AverageMetric
from crslab.evaluator.metrics.gen import PPLMetric
from crslab.system.base import BaseSystem
from crslab.system.utils.functions import ind2txt


class KGSFSystem(BaseSystem):
    """This is the system for KGSF model"""

    def __init__(self, opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data, restore_system=False,
                 interact=False, debug=False, tensorboard=False):
        """

        Args:
            opt (dict): Indicating the hyper parameters.
            train_dataloader (BaseDataLoader): Indicating the train dataloader of corresponding dataset.
            valid_dataloader (BaseDataLoader): Indicating the valid dataloader of corresponding dataset.
            test_dataloader (BaseDataLoader): Indicating the test dataloader of corresponding dataset.
            vocab (dict): Indicating the vocabulary.
            side_data (dict): Indicating the side data.
            restore_system (bool, optional): Indicating if we store system after training. Defaults to False.
            interact (bool, optional): Indicating if we interact with system. Defaults to False.
            debug (bool, optional): Indicating if we train in debug mode. Defaults to False.
            tensorboard (bool, optional) Indicating if we monitor the training performance in tensorboard. Defaults to False. 

        """
        super(KGSFSystem, self).__init__(opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data,
                                         restore_system, interact, debug, tensorboard)

        self.ind2tok = vocab['ind2tok']
        self.end_token_idx = vocab['end']
        self.item_ids = side_data['item_entity_ids']

        self.pretrain_optim_opt = self.opt['pretrain']
        self.rec_optim_opt = self.opt['rec']
        self.conv_optim_opt = self.opt['conv']
        self.pretrain_epoch = self.pretrain_optim_opt['epoch']
        self.rec_epoch = self.rec_optim_opt['epoch']
        self.conv_epoch = self.conv_optim_opt['epoch']
        self.pretrain_batch_size = self.pretrain_optim_opt['batch_size']
        self.rec_batch_size = self.rec_optim_opt['batch_size']
        self.conv_batch_size = self.conv_optim_opt['batch_size']

    def rec_evaluate(self, rec_predict, item_label):
        rec_predict = rec_predict.cpu()
        rec_predict = rec_predict[:, self.item_ids]
        _, rec_ranks = torch.topk(rec_predict, 50, dim=-1)
        rec_ranks = rec_ranks.tolist()
        item_label = item_label.tolist()
        for rec_rank, item in zip(rec_ranks, item_label):
            item = self.item_ids.index(item)
            self.evaluator.rec_evaluate(rec_rank, item)

    def conv_evaluate(self, prediction, response):
        prediction = prediction.tolist()
        response = response.tolist()
        for p, r in zip(prediction, response):
            p_str = ind2txt(p, self.ind2tok, self.end_token_idx)
            r_str = ind2txt(r, self.ind2tok, self.end_token_idx)
            self.evaluator.gen_evaluate(p_str, [r_str])

    def step(self, batch, stage, mode):
        batch = [ele.to(self.device) for ele in batch]
        if stage == 'pretrain':
            info_loss = self.model.forward(batch, stage, mode)
            if info_loss is not None:
                self.backward(info_loss.sum())
                info_loss = info_loss.sum().item()
                self.evaluator.optim_metrics.add("info_loss", AverageMetric(info_loss))
        elif stage == 'rec':
            rec_loss, info_loss, rec_predict = self.model.forward(batch, stage, mode)
            if info_loss:
                loss = rec_loss + 0.025 * info_loss
            else:
                loss = rec_loss
            if mode == "train":
                self.backward(loss.sum())
            else:
                self.rec_evaluate(rec_predict, batch[-1])
            rec_loss = rec_loss.sum().item()
            self.evaluator.optim_metrics.add("rec_loss", AverageMetric(rec_loss))
            if info_loss:
                info_loss = info_loss.sum().item()
                self.evaluator.optim_metrics.add("info_loss", AverageMetric(info_loss))
        elif stage == "conv":
            if mode != "test":
                gen_loss, pred = self.model.forward(batch, stage, mode)
                if mode == 'train':
                    self.backward(gen_loss.sum())
                else:
                    self.conv_evaluate(pred, batch[-1])
                gen_loss = gen_loss.sum().item()
                self.evaluator.optim_metrics.add("gen_loss", AverageMetric(gen_loss))
                self.evaluator.gen_metrics.add("ppl", PPLMetric(gen_loss))
            else:
                pred = self.model.forward(batch, stage, mode)
                self.conv_evaluate(pred, batch[-1])
        else:
            raise

    def pretrain(self):
        self.init_optim(self.pretrain_optim_opt, self.model.parameters())

        for epoch in range(self.pretrain_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Pretrain epoch {str(epoch)}]')
            for batch in self.train_dataloader.get_pretrain_data(self.pretrain_batch_size, shuffle=False):
                self.step(batch, stage="pretrain", mode='train')
            self.evaluator.report()

    def train_recommender(self):
        self.init_optim(self.rec_optim_opt, self.model.parameters())

        for epoch in range(self.rec_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Recommendation epoch {str(epoch)}]')
            logger.info('[Train]')
            for batch in self.train_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                self.step(batch, stage='rec', mode='train')
            self.evaluator.report(epoch=epoch, mode='train')
            # val
            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                    self.step(batch, stage='rec', mode='val')
                self.evaluator.report(epoch=epoch, mode='val')
                # early stop
                metric = self.evaluator.rec_metrics['hit@1'] + self.evaluator.rec_metrics['hit@50']
                if self.early_stop(metric):
                    break
        # test
        logger.info('[Test]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                self.step(batch, stage='rec', mode='test')
            self.evaluator.report(mode='test')

    def train_conversation(self):
        if os.environ["CUDA_VISIBLE_DEVICES"] == '-1':
            self.model.freeze_parameters()
        else:
            self.model.module.freeze_parameters()
        self.init_optim(self.conv_optim_opt, self.model.parameters())

        for epoch in range(self.conv_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Conversation epoch {str(epoch)}]')
            logger.info('[Train]')
            for batch in self.train_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                self.step(batch, stage='conv', mode='train')
            self.evaluator.report(epoch=epoch, mode='train')
            # val
            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                    self.step(batch, stage='conv', mode='val')
                self.evaluator.report(epoch=epoch, mode='val')
        # test
        logger.info('[Test]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                self.step(batch, stage='conv', mode='test')
            self.evaluator.report(mode='test')

    def fit(self):
        self.pretrain()
        self.train_recommender()
        self.train_conversation()

    def interact(self):
        self.init_interact()
        input_text = self.get_input("en")
        while not self.finished:
            #자연어로 된 input 처리해 tensor만듬
            KGSF_input = self.process_input(input_text, 'conv')

            #처리한 tensor 모델에 넣어 결과 받는다.
            preds = self.model.forward(KGSF_input, 'conv', 'test').tolist()[0]

            #모델이 출력한 결과 다시 자연어로 번역
            p_str = ind2txt(preds, self.ind2tok, self.end_token_idx)

            #여기서 시간 많이, 왜?
            token_ids, entity_ids, movie_ids, word_ids = self.convert_to_id(p_str, 'conv')
            
            #차후 필요하면 수정 
            self.update_context('conv', token_ids, entity_ids, movie_ids, word_ids)

            print(f"[Response]:\n{p_str}")
            input_text = self.get_input("en")

    def init_interact(self):
        self.finished = False
        self.context = {
            'rec': {},
            'conv': {}
        }
        for key in self.context:
            self.context[key]['context_tokens'] = []
            self.context[key]['response'] = []
            self.context[key]['context_entities'] = []
            self.context[key]['context_words'] = []
            self.context[key]['context_items'] = []
            self.context[key]['items'] = []
            self.context[key]['entity_set'] = set()
            self.context[key]['word_set'] = set()

    def process_input(self, input_text, stage):
        token_ids, entity_ids, movie_ids, word_ids = self.convert_to_id(input_text, stage)
        
        self.update_context(stage, token_ids, entity_ids, movie_ids, word_ids)
 
        data = {'role': 'Seeker', 'context_tokens': self.context[stage]['context_tokens'],
                'response': self.context[stage]['response'],
                'context_entities': self.context[stage]['context_entities'],
                'context_words': self.context[stage]['context_words'],
                'context_items': self.context[stage]['context_items'],
                'items': self.context[stage]['context_items']}

        dataloader = get_dataloader(self.opt, data, self.vocab) 

        if stage == 'conv':
            data = dataloader.conv_interact(data)

        data = [ele.to(self.device) if isinstance(ele, torch.Tensor) else ele for ele in data]
        return data

    def convert_to_id(self, text, stage):
        #token의 경우 text를 단어별로 분해한 것. ex: 'jack is having dinner' => ['jac','is','having','dinner']
        tokens = self.tokenize(text, 'nltk')    
        
        #임시
        """
        if tokens[0] == '__start__':
            del tokens[0]
        """

        #'i like the movie avengers' 입력하면 
        #entities:['<http://dbpedia.org/resource/Z_movie>', '<http://dbpedia.org/resource/Masked_Avengers_(1981_film)>']
        #words:['juliet', 'saintlike', 'buy_presents_for_others', 'movie', 'avengers']

        #token의 길이에 비례해 link 함수에서 엄청난 시간이 걸린다. 왜?
        entities = self.link(tokens, self.side_data['entity_kg']['entity'])
        words = self.link(tokens, self.side_data['word_kg']['entity'])

        token_ids = [self.vocab['tok2ind'].get(token, self.vocab['unk']) for token in tokens]
        entity_ids = [self.vocab['entity2id'][entity] for entity in entities if
                      entity in self.vocab['entity2id']]

        movie_ids = [entity_id for entity_id in entity_ids if entity_id in self.item_ids]
        word_ids = [self.vocab['word2id'][word] for word in words if word in self.vocab['word2id']]

        return token_ids, entity_ids, movie_ids, word_ids