import torchmetrics as tm
import pandas as pd
import datetime
import typer


class NetEvaluator:
    def __init__(self):
        evaluators = [tm.F1(multiclass=False),
                      tm.Accuracy(multiclass=False),
                      tm.Precision(multiclass=False),
                      tm.Recall(multiclass=False)]
        self.names = ['f1', 'acc', 'precision', 'recall']
        self.evaluators = {name: evaluator for name, evaluator in zip(self.names, evaluators)}
        self.metrics = {'f1': [], 'acc': [], 'precision': [], 'recall': []}
        self.cols = ['batch_num', 'f1', 'accuracy', 'recall', 'precision', 'loss']
        self.score = pd.DataFrame(columns=self.cols)

    def update(self, i, pred, target, loss, device, prints=True):
        score_line = [i]
        for met in self.evaluators.values():
            met.to(device)
            met.update(target, pred)
            score_line.append(met.compute())
        score_line.append(loss)
        score_line[1:] = [float(i.cpu().numpy()) for i in score_line[1: ]]
        self.score.loc[i] = score_line
        if prints:
            ct = datetime.datetime.now()
            ts = ct.timestamp()
            me = f'{ts} - Scores: f1: {score_line[0]} \tacc: {score_line[1]}' \
                 f'\t precision: {score_line[2]} \t recall: {score_line[3]} \t auc: {score_line[4]} \t'\
                 f'\t loss: {score_line[5]}'
            # print(me)
        return score_line

    def save(self,
             filename: str,
             ts: str):
        filename = f'test_df_{filename}_{ts}.csv'
        save_loc = f'results/{filename}'
        self.score.to_csv(save_loc)
        typer.echo(f'Test DataFrame saved to {save_loc}')
        return None
