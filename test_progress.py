import typer
import time

length = 1000
with typer.progressbar(length=length) as progress:
    time.sleep(2)
    progress.update(250)
    progress.label = "Processing"
    time.sleep(2)
    progress.update(250)
    progress.label = "done with kaka pip"
    time.sleep(2)
    progress.update(250)
    time.sleep(2)
    progress.label = "Done"
    progress.update(250)
    typer.secho(f"\nProcessed {length} things in batches.")