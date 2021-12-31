import torchmetrics as tm
import pandas as pd
import datetime
import typer


class NetEvaluator:
    def __init__(self):
        evaluators = [tm.F1(average='macro', num_classes=2),
                      tm.Accuracy(num_classes=2),
                      tm.Precision(average='macro', num_classes=2),
                      tm.Recall(average='macro', num_classes=2)]
                      # tm.AUROC(average='macro', num_classes=2)]
        self.names = ['f1', 'acc', 'precision', 'recall', 'auc']
        self.evaluators = {name: evaluator for name, evaluator in zip(self.names, evaluators)}
        self.metrics = {'f1': [], 'acc': [], 'precision': [], 'recall': [], 'auc': []}
        self.cols = ['batch_num', 'f1', 'accuracy', 'recall', 'precision', 'AUC']
        self.score = pd.DataFrame(columns=self.cols)

    def update(self, i, pred, target, prints=True):
        score_line = []
        for met in self.evaluators.values():
            met.update(target, pred)
            score_line.append(met.compute())
        self.score.loc[i] = score_line
        if prints:
            ct = datetime.datetime.now()
            ts = ct.timestamp()
            me = f'{ts} - Scores: f1: {score_line[0]} \tacc: {score_line[1]}' \
                 f'\t precision: {score_line[2]} \t recall: {score_line[3]} \t auc: {score_line[4]} \t'
            print(me)
        return score_line

    def __repr__(self):
        print(self.score)
        return self.score

    def save(self):
        self.score.to_csv(f'results/{self.score.name}.csv')
        typer.echo(f'Test DataFrame saved to {self.score}')
        return None
