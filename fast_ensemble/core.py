from pytorch import torch


class Ensemble:
    def __init__(self, dls, models: dict, vocab: list = [0, 1]):
        self.models = models
        self.vocab = vocab
        self.dls = dls
        self.model_list = models.values()
        print(f"vocab: {self.vocab}")
        for name, _ in models.items():
            print(f"loaded: {name}")

    def calc_probas(self, item):
        probas = []
        for _, model in self.models.items():
            _, _, p = model.predict(item)
            probas.append(p)

        probas = torch.stack(probas, dim=0)
        return probas

    def predict(self, item):
        probas = self.calc_probas(item)
        mean, std = probas.mean(axis=0), probas.std(axis=0)

        return self.vocab[mean.argmax()], mean.argmax(), std

    def get_preds(
        self,
        ds_idx=1,
        dl=None,
        with_input=False,
        with_decoded=False,
        with_loss=False,
        act=None,
        inner=False,
        reorder=True,
        cbs=None,
        **kwargs,
    ):

        if dl is None:
            dl = self.dls[1]

        predictions = []
        losses = []
        res = []

        for name, model in self.models.items():
            print(f"Getting predictions from {name} \n")
            inputs, preds, targs, decoded, loss = model.get_preds(
                dl=self.dls.valid, with_input=True, with_loss=True, with_decoded=True
            )
            predictions.append(preds)
            losses.append(loss)

        preds = torch.stack(predictions).mean(0)
        decoded = preds.argmax(1)

        if with_input:
            res.append(inputs)

        res.append(preds)
        res.append(targs)

        if with_decoded:
            res.append(decoded)

        if with_loss:
            res.append(torch.stack(losses, dim=1).mean(1))

        return tuple(res)

    def calc_metrics(self, metrics: dict):
        res = {}
        predictions, targs, decoded, losses = self.get_preds(
            dl=self.dls.valid, with_input=False, with_loss=True, with_decoded=True
        )
        for name, metric in metrics.items():
            res[name] = metric(decoded, targs)
        return res
