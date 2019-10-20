import torch
import logging
import sys

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)


def train(model, train_iter, val_iter, test_iter, optimizer, scheduler,
          max_epochs=10):
    model.train()

    max_f1 = 0
    test_f1 = 0
    for epoch_idx in range(max_epochs):
        logging.info("Started epoch %d" % (epoch_idx + 1))
        for idx, batch_data in enumerate(train_iter):
            optimizer.zero_grad()
            loss = model(batch_data)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0)

            optimizer.step()

        logging.info("Epoch done!\n")
        f1 = eval(model, val_iter)
        if f1 > max_f1:
            max_f1 = f1
            test_f1 = eval(model, test_iter)
            logging.info("Max F1: %.3f" % max_f1)
        logging.info(
            "Val F1: %.3f Epoch: %d (Max F1: %.3f)" %
            (f1, epoch_idx + 1, max_f1))
        # Scheduler step
        scheduler.step(f1)

        sys.stdout.flush()

    logging.info("Training done!\n")
    logging.info("Val F1 %.3f, Test F1 %.3f" % (max_f1, test_f1))
    return (max_f1, test_f1)


def eval(model, val_iter):
    model.eval()
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    with torch.no_grad():
        for batch_data in val_iter:
            label = batch_data.label.cuda().float()
            _, pred = model(batch_data)
            pred = (pred > 0.5).float()

            tp += torch.sum(label * pred)
            tn += torch.sum((1 - label) * (1 - pred))
            fp += torch.sum((1 - label) * pred)
            fn += torch.sum(label * (1 - pred))

    if tp > 0:
        recall = tp/(tp + fn)
        precision = tp/(tp + fp)

        f_score = (2 * recall * precision) / (recall + precision)
    else:
        f_score = 0.0

    model.train()
    return f_score
