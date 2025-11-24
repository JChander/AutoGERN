from sklearn.metrics import roc_auc_score
from src.utils_LP import *

criterion = torch.nn.functional.cross_entropy


def search(model, dataloaders, args, logger):
    device = get_device(args)
    model.to(device)
    train_loader, val_loader, test_loader = dataloaders
    optimizer = get_optimizer(model, args)
    # for name, param in model.named_parameters():
    #     print(name, '\t\t', param.shape)

    metric = args.metric
    recorder = SearchRecorder(metric)
    for step in range(args.epoch):
        # print(model.log_alpha_agg)
        # print(model.Z_agg_hard)
        # if step < 2:
        #     print('#########################################################################################')
        #     for n, p in model.named_parameters():
        #         print(n)
        #         print(p)

        optimize_model(model, train_loader, optimizer, device, args)
        train_loss, train_acc, train_auc = eval_model(model, train_loader, device)
        val_loss, val_acc, val_auc = eval_model(model, val_loader, device)
        #         test_loss, test_acc, test_auc = eval_model(model, test_loader, device)
        #         recorder.update(train_acc, train_auc, val_acc, val_auc, test_acc, test_auc)
        #####################################################################################################
        #####################################################################################################
        model.update_z_hard()
        if step > 30 and step % 5 == 0 and model.temperature >= 1e-20:
            model.temperature *= 1e-1
            # model.temperature /= 1.1
        #####################################################################################################
        #####################################################################################################

        recorder.update(step, train_acc, train_auc, val_acc, val_auc)

        logger.info('epoch %d best val %s: %.4f, train loss: %.4f; train %s: %.4f val %s: %.4f' %
                    (step, metric, recorder.get_best_metric()[0], train_loss,
                     metric, train_auc,
                     metric, val_auc))
    #     logger.info('(With validation) final test %s: %.4f (epoch: %d, val %s: %.4f)' %
    #                 (metric, recorder.get_best_metric(val=True)[0],
    #                  recorder.get_best_metric(val=True)[1], metric, recorder.get_best_val_metric(val=True)[0]))
    logger.info('(Search Stage) best val acc: %.4f (epoch: %d) ' % recorder.get_best_acc())
    logger.info('(Search Stage) best val auc: %.4f (epoch: %d) ' % recorder.get_best_auc())

    results, max_step = recorder.get_best_metric()
    model.max_step = max_step
    model.best_metric_search = results
    return model


def retrain(model, dataloaders, args, logger):
    device = get_device(args)
    model.derive_arch()

    logger.info('Derived z')
    logger.info(model.searched_arch_z)
    logger.info('Derived arch')
    logger.info(model.searched_arch_op)

    def weight_reset(m):
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()
    model.apply(weight_reset)
    model.to(device)

    train_loader, val_loader, test_loader = dataloaders
    optimizer = get_optimizer(model, args)
    metric = args.metric
    #     recorder = Recorder(metric)
    recorder = RetrainRecorder(metric)
    for step in range(args.retrain_epoch):
        optimize_model(model, train_loader, optimizer, device, args)
        train_loss, train_acc, train_auc = eval_model(model, train_loader, device)
        #         val_loss, val_acc, val_auc = eval_model(model, val_loader, device)
        test_loss, test_acc, test_auc, test_labels, test_predictions = eval_model(model, test_loader, device, return_predictions=True)
        #         recorder.update(train_acc, train_auc, val_acc, val_auc, test_acc, test_auc)
        #         recorder.update(train_acc, train_auc, val_acc, val_auc)
        recorder.update(step, train_acc, train_auc, test_acc, test_auc, test_labels, test_predictions)

        logger.info('epoch %d best test %s: %.4f, retrain loss: %.4f; retrain %s: %.4f test %s: %.4f' %
                    (step, metric, recorder.get_best_metric()[0], train_loss,
                     metric, train_auc,
                     metric, test_auc))
    #     logger.info('(With validation) final test %s: %.4f (epoch: %d, val %s: %.4f)' %
    #                 (metric, recorder.get_best_metric(val=True)[0],
    #                  recorder.get_best_metric(val=True)[1], metric, recorder.get_best_val_metric(val=True)[0]))
    logger.info('(Retrain Stage) best test acc: %.4f (epoch: %d) ' % recorder.get_best_acc())
    logger.info('(Retrain Stage) best test auc: %.4f (epoch: %d) ' % recorder.get_best_auc())

    #     return recorder.get_best_metric()[0], recorder.get_best_metric()[0]
    # return recorder.get_best_metric()[0]
    best_metric, max_step = recorder.get_best_metric()
    model.max_step = max_step
    model.best_metric_retrain = best_metric
    test_predictions, test_labels = recorder.get_best_predicitons()
    return model, test_predictions.cpu().detach().numpy(), test_labels.cpu().detach().numpy()


def optimize_model(model, dataloader, optimizer, device, args):
    model.train()
    # setting of data shuffling move to dataloader creation
    for batch in dataloader:
        batch = batch.to(device)
        label = batch.y
        prediction = model(batch)
        loss = criterion(prediction, label, reduction='mean')
        # loss.backward()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
        optimizer.step()


def eval_model(model, dataloader, device, return_predictions=False):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            labels.append(batch.y)
            prediction = model(batch)
            predictions.append(prediction)
        predictions = torch.cat(predictions, dim=0)
        labels = torch.cat(labels, dim=0)
    loss, acc, auc = compute_metric(predictions, labels)
    if not return_predictions:
        return loss, acc, auc
    else:
        return loss, acc, auc, labels, predictions


def compute_metric(predictions, labels):
    with torch.no_grad():
        # compute loss:
        loss = criterion(predictions, labels, reduction='mean').item()
        # compute acc:
        correct_predictions = (torch.argmax(predictions, dim=1) == labels)
        acc = correct_predictions.sum().cpu().item()/labels.shape[0]
        # compute auc:
        predictions = torch.nn.functional.softmax(predictions, dim=-1)
        multi_class = 'ovr'
        if predictions.size(1) == 2:
            predictions = predictions[:, 1]
            multi_class = 'raise'
        auc = roc_auc_score(labels.cpu().numpy(), predictions.cpu().numpy(), multi_class=multi_class)
    return loss, acc, auc


class SearchRecorder:
    """
    always return test numbers except the last method
    """

    def __init__(self, metric):
        self.metric = metric
        self.max_step = 0
        self.train_acc, self.val_acc, self.train_auc, self.val_auc = 0., 0., 0., 0.

    def update(self, step, train_acc, train_auc, val_acc, val_auc):
        if self.val_auc < val_auc:
            self.train_acc=train_acc
            self.train_auc=train_auc
            self.val_acc=val_acc
            self.val_auc=val_auc
            self.max_step=step


    def get_best_metric(self):
        dic = {'acc': self.get_best_acc(), 'auc': self.get_best_auc()}
        return dic[self.metric]

    def get_best_acc(self):
        #         if val:
        #             max_step = int(np.argmax(np.array(self.val_accs)))
        #         else:
        #             max_step = int(np.argmax(np.array(self.test_accs)))
        #         return self.test_accs[max_step], max_step
        return self.val_acc, self.max_step

    def get_best_auc(self):
        #         if val:
        #             max_step = int(np.argmax(np.array(self.val_aucs)))
        #         else:
        #             max_step = int(np.argmax(np.array(self.test_aucs)))
        #         return self.test_aucs[max_step], max_step
        return self.val_auc, self.max_step


class RetrainRecorder:
    """
    always return test numbers except the last method
    """

    def __init__(self, metric):
        self.metric = metric
        self.max_step = 0
        self.train_acc, self.test_acc, self.train_auc, self.test_auc = 0., 0., 0., 0.
        self.test_labels, self.test_predictions = [], []

    def update(self, step, train_acc, train_auc, test_acc, test_auc, test_labels, test_predictions):
        if self.test_auc < test_auc:
            self.train_acc=train_acc
            self.train_auc=train_auc
            self.test_acc=test_acc
            self.test_auc=test_auc
            self.max_step = step
            self.test_labels = test_labels
            self.test_predictions = test_predictions

    def get_best_metric(self):
        dic = {'acc': self.get_best_acc(), 'auc': self.get_best_auc()}
        return dic[self.metric]

    def get_best_acc(self):
        #         if val:
        #             max_step = int(np.argmax(np.array(self.val_accs)))
        #         else:
        #             max_step = int(np.argmax(np.array(self.test_accs)))
        #         return self.test_accs[max_step], max_step
        #         max_step = int(np.argmax(np.array(self.val_accs)))
        return self.test_acc, self.max_step

    def get_best_auc(self):
        #         if val:
        #             max_step = int(np.argmax(np.array(self.val_aucs)))
        #         else:
        #             max_step = int(np.argmax(np.array(self.test_aucs)))
        #         return self.test_aucs[max_step], max_step
        #         max_step = int(np.argmax(np.array(self.val_aucs)))
        return self.test_auc, self.max_step

    def get_best_predicitons(self):
        return self.test_predictions, self.test_labels