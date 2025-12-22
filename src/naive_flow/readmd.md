
# NaiveFlow

[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)

NaiveFlow, a lightweight and unobtructive higher level framework based on Pytorch

## Why NaiveFlow

NaiveFlow follows a philosophy similar to that of TensorBoard's SummaryWriter, allowing for easy integration into your code without affecting the original framework.

The `SimpleTracker` takes a PyTorch model (`torch.nn.Module`), an optimizer, and an optional scheduler as inputs, saving and loading them in an elegant manner.


### Tracker

Tracker is a simple utility that handles checkpoint saving, early stopping, and integrate with tensorboard.SummaryWriter.


```python
from torch.utils.tensorboard import SummaryWriter
import naive_flow as nf

tracker_config = nf.tracker.TrackerConfig(
    epochs_per_checkpoint=0,
    enable_logging=True,
    save_n_best=0,
    early_stopping_rounds=5,
    save_end=False,
    comment="_NMIST",
)

tracker = nf.tracker.SimpleTracker(
    model,
    optimizer,
    scheduler,
    **dict(tracker_config),
)

"""
Initialize summary writer and register the writer.
This can be created with a shorthand writer = tracker.create_summary_writer()
"""
writer = SummaryWriter(
    log_dir=tracker.log_dir, purge_step=tracker.start_epoch
)
tracker.register_summary_writer(writer)

"""Register scalars that would be used in experiment.
Passing for_early_stopping=True allows tracker to know when to quit after no improvement of the
specified scalar.

For scalars not used for early stopping, it accepts Unix shell-style wildcards, which would be parsed using `fnmatch`.
"""
tracker.register_scalar(
    'loss/val', scalar_type='loss', for_early_stopping=True
)
tracker.register_scalar('loss/*', scalar_type='loss')
tracker.register_scalar('accuracy/*', scalar_type='accuracy')

for epoch in tracker.range(20): # <-- tracker.range instead of range.

    train_loss, train_acc = train_epoch(
        model, optimizer, scheduler, train_loader
    )
    writer.add_scalar('loss/train', train_loss, epoch)
    writer.add_scalar('accuracy/train', train_acc, epoch)

    val_loss, val_acc = eval_epoch(model, val_loader)
    writer.add_scalar('loss/val', val_loss, epoch)
    # Once the writer registered and 'loss/val' is registered for early stopping.
    # `tracker` would decide whether to quit the loop after early_stopping_rounds of no improvement
    writer.add_scalar('accuracy/val', val_acc, epoch)

    test_loss, test_acc = eval_epoch(model, test_loader)
    writer.add_scalar('loss/eval', test_loss, epoch)
    writer.add_scalar('accuracy/eval', test_acc, epoch)
    """Model, optimizer, and scheduler are saved based on the tracker_config automatically
    at the end on the for loop.
    """

print('results: ', tracker.get_best_scalars())
```